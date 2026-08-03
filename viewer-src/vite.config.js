import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    proxy: {
      '/mri': 'http://localhost:8765',
      '/api': 'http://localhost:8765',
    },
  },
  build: {
    outDir: '../public',
    emptyOutDir: true,
  },
});
