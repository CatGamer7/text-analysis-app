const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn, execSync } = require('child_process');
const http = require('http');

let mainWindow = null;
let backendProcess = null;
const BACKEND_PORT = 8000;
const LOG_FILE = path.join(app.getPath('temp'), 'text-analysis-studio-backend.log');

// Ensure only one instance runs at a time
const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

function logMsg(msg) {
  const line = `[${new Date().toISOString()}] ${msg}\n`;
  try {
    fs.appendFileSync(LOG_FILE, line);
  } catch (e) {}
  console.log(msg);
}

function getBackendExecutablePath() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'backend', 'text-analysis-backend.exe');
  } else {
    return path.join(__dirname, '..', 'dist', 'text-analysis-backend', 'text-analysis-backend.exe');
  }
}

function pingBackendOnce() {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${BACKEND_PORT}/api/v1/health`, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on('error', () => resolve(false));
    req.setTimeout(1000, () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function ensureBackendRunning() {
  // Check if already running (e.g. from dev session)
  const alreadyUp = await pingBackendOnce();
  if (alreadyUp) {
    logMsg('[Electron Main] Backend already active on port 8000, reusing.');
    return true;
  }

  const exePath = getBackendExecutablePath();
  logMsg(`[Electron Main] Launching backend from: ${exePath}`);

  if (!fs.existsSync(exePath)) {
    logMsg(`[Electron Main] ERROR: Executable not found at ${exePath}`);
    return false;
  }

  try {
    // Reset log file
    try { fs.writeFileSync(LOG_FILE, ''); } catch (e) {}

    backendProcess = spawn(exePath, [], {
      cwd: path.dirname(exePath),
      detached: false,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    backendProcess.on('error', (err) => {
      logMsg(`[Backend spawn error]: ${err.message}`);
    });

    backendProcess.stdout.on('data', (data) => {
      logMsg(`[Backend stdout]: ${data.toString().trim()}`);
    });

    backendProcess.stderr.on('data', (data) => {
      logMsg(`[Backend stderr]: ${data.toString().trim()}`);
    });

    backendProcess.on('close', (code) => {
      logMsg(`[Backend process exited with code ${code}]`);
      backendProcess = null;
    });

    // Wait up to 15 seconds for healthcheck
    for (let i = 0; i < 30; i++) {
      const ok = await pingBackendOnce();
      if (ok) {
        logMsg('[Electron Main] Backend healthcheck passed!');
        return true;
      }
      if (!backendProcess) {
        logMsg('[Electron Main] Backend process terminated prematurely.');
        return false;
      }
      await new Promise((r) => setTimeout(r, 500));
    }

    logMsg('[Electron Main] Backend healthcheck timeout.');
    return false;
  } catch (err) {
    logMsg(`[Electron Main] Exception spawning backend: ${err.message}`);
    return false;
  }
}

function killBackend() {
  if (backendProcess && backendProcess.pid) {
    const pid = backendProcess.pid;
    logMsg(`[Electron Main] Stopping backend process (PID: ${pid})`);
    try {
      if (process.platform === 'win32') {
        execSync(`taskkill /pid ${pid} /T /F`, { stdio: 'ignore' });
      } else {
        backendProcess.kill('SIGTERM');
      }
    } catch (e) {
      try { backendProcess.kill(); } catch (_) {}
    }
    backendProcess = null;
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1240,
    height: 840,
    minWidth: 1000,
    minHeight: 700,
    backgroundColor: '#07101B',
    title: 'Text Analysis Studio',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    },
    autoHideMenuBar: true,
    show: false
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

if (gotTheLock) {
  app.whenReady().then(async () => {
    createWindow();
    const ok = await ensureBackendRunning();
    if (!ok && mainWindow) {
      logMsg('[Electron Main] Backend failed to become healthy.');
    }
  });

  app.on('window-all-closed', () => {
    killBackend();
    if (process.platform !== 'darwin') {
      app.quit();
    }
  });

  app.on('before-quit', () => {
    killBackend();
  });
}
