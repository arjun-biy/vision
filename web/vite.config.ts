import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:5000",
      "/status": "http://127.0.0.1:5000",
      "/upload": "http://127.0.0.1:5000",
      "/outputs": "http://127.0.0.1:5000",
      "/result-text": "http://127.0.0.1:5000",
    },
  },
});
