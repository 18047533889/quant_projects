import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

// Vite + Vitest config for platform_web.
// The app is a pure static build: `npm run build` emits `dist/` (no backend
// bundling). Tests run in a `node` environment because every test targets the
// PURE helpers in src/lib (state classification + formatters) — no DOM needed.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
