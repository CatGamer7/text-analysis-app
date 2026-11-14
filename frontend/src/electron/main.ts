import { app, BrowserWindow } from "electron";
import path from "path";

const isDev = process.env.NODE_ENV === "development";
// Должен совпадать с server.port в vite.config.ts
const DEV_SERVER_URL = "http://localhost:5123";

function createMainWindow() {
  const mainWindow = new BrowserWindow({
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
    // Vite может стартовать позже Electron — пробуем снова, пока сервер не поднимется
    mainWindow.webContents.on("did-fail-load", () => {
      setTimeout(() => mainWindow.loadURL(DEV_SERVER_URL), 500);
    });
    mainWindow.loadURL(DEV_SERVER_URL);
  } else {
    mainWindow.removeMenu();
    mainWindow.loadFile(path.join(app.getAppPath(), "dist-react", "index.html"));
  }
}

app.whenReady().then(() => {
  createMainWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
