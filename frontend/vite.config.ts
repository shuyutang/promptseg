import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The build lands in backend/static, so `uvicorn app:app` serves the finished
// app with no node runtime involved -- node is only needed to rebuild it.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: '../backend/static',
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    // `npm run dev` gives hot reload against a backend on 8000.
    proxy: Object.fromEntries(
      ['/upload', '/workspaces', '/images', '/annotations', '/segment', '/frame.png',
       '/frame_info', '/labels', '/export.json', '/export.zip', '/health', '/dicom']
        .map((p) => [p, { target: 'http://127.0.0.1:8000', changeOrigin: true }]),
    ),
  },
})
