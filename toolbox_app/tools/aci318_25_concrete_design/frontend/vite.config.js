import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * Notes:
 * - The production build is intended to be served by your PySide6/FastAPI host (offline).
 * - We keep all API calls relative (e.g., /api/solve) so the plugin host can route them.
 * - For local development, Vite proxies /api to a backend running on localhost:8000.
 */
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true
      }
    }
  },
  build: {
    // Keep output straightforward for your host to copy/serve.
    outDir: "dist",
    sourcemap: true
  }
});
