/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the platform FastAPI service, e.g. `https://platform.example.com/api/v1`. */
  readonly VITE_PLATFORM_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
