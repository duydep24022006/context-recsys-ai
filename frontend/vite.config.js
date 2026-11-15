import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 8080,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET || 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/recommend': { target: process.env.VITE_API_TARGET || 'http://localhost:8000', changeOrigin: true },
      '/health': { target: process.env.VITE_API_TARGET || 'http://localhost:8000', changeOrigin: true },
      '/model': { target: process.env.VITE_API_TARGET || 'http://localhost:8000', changeOrigin: true },
      '/models': { target: process.env.VITE_API_TARGET || 'http://localhost:8000', changeOrigin: true },
      '/interactions': { target: process.env.VITE_API_TARGET || 'http://localhost:8000', changeOrigin: true },
      '/analytics': { target: process.env.VITE_API_TARGET || 'http://localhost:8000', changeOrigin: true },
      '/users': { target: process.env.VITE_API_TARGET || 'http://localhost:8000', changeOrigin: true },
      '/products': { target: process.env.VITE_API_TARGET || 'http://localhost:8000', changeOrigin: true },
      '/admin': { target: process.env.VITE_API_TARGET || 'http://localhost:8000', changeOrigin: true }
    }
  }
})
