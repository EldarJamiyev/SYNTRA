import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { syntraLocalApiPlugin } from './scripts/vite-local-api.mjs';

export default defineConfig({
  plugins: [react(), syntraLocalApiPlugin()],
  server: {
    host: '127.0.0.1',
    port: 3000,
    strictPort: true,
    proxy: {
      '/local-api': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: false,
      },
    },
  },
});
