import type { NextConfig } from "next";

/**
 * The browser always talks same-origin to `/api/*`; Next proxies those requests
 * to the FastAPI backend. Override the target with NEURALSCOPE_API_URL.
 */
const API_URL = (process.env.NEURALSCOPE_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${API_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
