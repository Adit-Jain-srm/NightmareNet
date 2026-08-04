import type { NextConfig } from "next";

/** When the browser uses same-origin fetches to `/api/...`, proxy to the FastAPI app. */
const apiRewriteBase =
  process.env.NEXT_API_REWRITE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

const cspHeader = `
  default-src 'self';
  script-src 'self' 'unsafe-eval' 'unsafe-inline';
  style-src 'self' 'unsafe-inline';
  img-src 'self' blob: data:;
  font-src 'self';
  object-src 'none';
  base-uri 'self';
  form-action 'self';
  frame-ancestors 'none';
`.replace(/\s{2,}/g, " ").trim();

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          {
            key: "Content-Security-Policy",
            value: cspHeader,
          },
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
        ],
      },
    ];
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
