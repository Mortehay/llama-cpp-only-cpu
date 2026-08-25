import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The build is emitted INTO the FastAPI static tree, so production stays one
// container serving one origin. No second web server, no CORS, no separate
// deployment step - `npm run build` is the whole thing.
//
// `base` must match where FastAPI mounts it (/static is already a StaticFiles
// mount), otherwise the built index.html asks for /assets/... at the root and
// gets the SPA shell back instead of JavaScript.
export default defineConfig({
  plugins: [react()],
  base: '/static/app/',
  build: {
    outDir: '../src/sprite_generator/static/app',
    emptyOutDir: true,
    // The Python image is built from the repo, so a source map here would ship
    // to anything that can reach the port. Off deliberately.
    sourcemap: false,
  },
  server: {
    host: true,
    port: 5173,
    // Dev only. `npm run dev` serves the UI with hot reload and forwards
    // everything the API owns to the real service, so the dev UI talks to real
    // jobs and real GPU work rather than a mock.
    proxy: {
      '/api': 'http://localhost:8001',
      '/images': 'http://localhost:8001',
      '/static': 'http://localhost:8001',
    },
  },
})
