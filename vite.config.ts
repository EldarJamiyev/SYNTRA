import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { syntraLocalApiPlugin } from './scripts/vite-local-api.mjs';

const loopbackHosts = ['localhost', '127.0.0.1'];

export default defineConfig({
  plugins: [react(), syntraLocalApiPlugin()],
  server: {
    host: '127.0.0.1',
    port: 3000,
    strictPort: true,
    allowedHosts: loopbackHosts,
    proxy: {
      '/local-api': {
        target: 'http://127.0.0.1:3001',
        changeOrigin: true,
        headers: {
          origin: 'http://127.0.0.1:3000',
        },
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setHeader('origin', 'http://127.0.0.1:3000');
            proxyReq.setHeader('sec-fetch-site', 'same-origin');
          });
        },
      },
    },
  },
  preview: {
    host: '127.0.0.1',
    port: 3000,
    strictPort: true,
    allowedHosts: loopbackHosts,
  },
});
