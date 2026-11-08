const { app, BrowserWindow, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

let mainWindow = null;
let backendProcess = null;
const BACKEND_PORT = 8000;

function getBackendExecutablePath() {
  if (app.isPackaged) {
    // In packaged app, extraResources maps to process.resourcesPath/backend/text-analysis-backend.exe
    return path.join(process.resourcesPath, 'backend', 'text-analysis-backend.exe');
  } else {
    // In development mode
    return path.join(__dirname, '..', 'dist', 'text-analysis-backend', 'text-analysis-backend.exe');
  }
}

function startBackend() {
  const exePath = getBackendExecutablePath();
  console.log('[Electron Main] Launching backend from:', exePath);

  try {
    backendProcess = spawn(exePath, [], {
      detached: false,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    backendProcess.stdout.on('data', (data) => {
      console.log(`[Backend stdout]: ${data}`);
    });

    backendProcess.stderr.on('data', (data) => {
      console.error(`[Backend stderr]: ${data}`);
    });

    backendProcess.on('close', (code) => {
      console.log(`[Backend process exited with code ${code}]`);
    });
  } catch (err) {
    console.error('[Electron Main] Failed to start backend:', err);
  }
}

function checkBackendHealth(retries = 30, delay = 500) {
  return new Promise((resolve) => {
    let attempts = 0;
    const interval = setInterval(() => {
      attempts++;
      http.get(`http://127.0.0.1:${BACKEND_PORT}/api/v1/health`, (res) => {
        if (res.statusCode === 200) {
          clearInterval(interval);
          resolve(true);
        }
      }).on('error', () => {
        if (attempts >= retries) {
          clearInterval(interval);
          resolve(false);
        }
      });
    }, delay);
  });
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

app.whenReady().then(async () => {
  startBackend();
  createWindow();

  const isHealthy = await checkBackendHealth();
  if (!isHealthy) {
    console.warn('[Electron Main] Backend healthcheck timeout, proceeding anyway.');
  }
});

app.on('window-all-closed', () => {
  if (backendProcess) {
    try {
      backendProcess.kill();
    } catch (e) {
      console.error('Error killing backend:', e);
    }
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (backendProcess) {
    try {
      backendProcess.kill();
    } catch (e) {}
  }
});
