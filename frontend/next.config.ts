import type { NextConfig } from "next";

/** When the browser uses same-origin fetches to `/api/...`, proxy to the FastAPI app. */
const apiRewriteBase =
  process.env.NEXT_API_REWRITE_URL?.replace(/\/$/, "") ||
  "http://127.0.0.1:8000";

// Note: 'unsafe-inline' required by Next.js for inline script hydration.
// To remove it, enable strict CSP with nonce: https://nextjs.org/docs/app/building-your-application/configuring/content-security-policy
const apiOrigin = process.env.NEXT_PUBLIC_API_URL
  ? new URL(process.env.NEXT_PUBLIC_API_URL).origin
  : "";

const sentryConnectOrigin = (() => {
  const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN?.trim();
  if (!dsn) return "";
  const match = dsn.match(/@([^/]+)/);
  return match ? ` https://${match[1]}` : "";
})();

const cspHeader = `
  default-src 'self';
  script-src 'self' 'unsafe-eval' 'unsafe-inline';
  style-src 'self' 'unsafe-inline';
  img-src 'self' blob: data:;
  font-src 'self';
  connect-src 'self'${apiOrigin ? ` ${apiOrigin}` : ""}${sentryConnectOrigin};
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
