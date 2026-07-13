import type { NextConfig } from "next";

/** When the browser uses same-origin fetches to `/api/...`, proxy to the FastAPI app. */
const apiRewriteBase =
  process.env.NEXT_API_REWRITE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  turbopack: {
    root: __dirname,
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiRewriteBase}/api/:path*`,
      },
      {
        source: "/ws/:path*",
        destination: `${apiRewriteBase}/ws/:path*`,
      },
    ];
  },
};

export default nextConfig;
