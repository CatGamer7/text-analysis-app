import { app, BrowserWindow } from "electron";
import path from "path";
import fs from "fs";
import http from "http";
import { spawn, execSync, ChildProcess } from "child_process";

const isDev = process.env.NODE_ENV === "development";
const DEV_SERVER_URL = "http://localhost:5123";
const BACKEND_PORT = 8000;
const LOG_FILE = path.join(app.getPath("temp"), "text-analysis-backend-electron.log");

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;

// Single instance lock
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

function logMsg(msg: string) {
  const line = `[${new Date().toISOString()}] ${msg}\n`;
  try {
    fs.appendFileSync(LOG_FILE, line);
  } catch {}
  console.log(msg);
}

function getBackendExecutablePath(): string {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "backend", "text-analysis-backend.exe");
  } else {
    return path.join(app.getAppPath(), "..", "dist", "text-analysis-backend", "text-analysis-backend.exe");
  }
}

function pingBackendOnce(): Promise<boolean> {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${BACKEND_PORT}/api/v1/health`, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.setTimeout(1000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function ensureBackendRunning(): Promise<boolean> {
  const alreadyUp = await pingBackendOnce();
  if (alreadyUp) {
    logMsg("[Electron Main] Backend already running on port 8000.");
    return true;
  }

  const exePath = getBackendExecutablePath();
  logMsg(`[Electron Main] Launching backend from: ${exePath}`);

  if (!fs.existsSync(exePath)) {
    logMsg(`[Electron Main] WARNING: Backend executable not found at ${exePath}`);
    return false;
  }

  try {
    backendProcess = spawn(exePath, [], {
      cwd: path.dirname(exePath),
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });

    backendProcess.stdout?.on("data", (data) => {
      logMsg(`[Backend stdout]: ${data.toString().trim()}`);
    });

    backendProcess.stderr?.on("data", (data) => {
      logMsg(`[Backend stderr]: ${data.toString().trim()}`);
    });

    backendProcess.on("close", (code) => {
      logMsg(`[Backend exited with code ${code}]`);
      backendProcess = null;
    });

    for (let i = 0; i < 30; i++) {
      const ok = await pingBackendOnce();
      if (ok) {
        logMsg("[Electron Main] Backend is healthy!");
        return true;
      }
      if (!backendProcess) break;
      await new Promise((r) => setTimeout(r, 500));
    }
  } catch (err) {
    logMsg(`[Electron Main] Error launching backend: ${err}`);
  }

  return false;
}

function stopBackendProcess() {
  if (backendProcess && backendProcess.pid) {
    logMsg(`[Electron Main] Stopping backend PID ${backendProcess.pid}`);
    try {
      if (process.platform === "win32") {
        execSync(`taskkill /pid ${backendProcess.pid} /T /F`);
      } else {
        backendProcess.kill("SIGTERM");
      }
    } catch {}
    backendProcess = null;
  }
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 960,
    minHeight: 600,
    title: "Text Analysis",
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (isDev) {
    mainWindow.webContents.on("did-fail-load", () => {
      setTimeout(() => mainWindow?.loadURL(DEV_SERVER_URL), 500);
    });
    mainWindow.loadURL(DEV_SERVER_URL);
  } else {
    mainWindow.removeMenu();
    mainWindow.loadFile(path.join(app.getAppPath(), "dist-react", "index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(async () => {
  ensureBackendRunning();
  createMainWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
  });
});

app.on("before-quit", () => {
  stopBackendProcess();
});

app.on("window-all-closed", () => {
  stopBackendProcess();
  if (process.platform !== "darwin") {
    app.quit();
  }
});
