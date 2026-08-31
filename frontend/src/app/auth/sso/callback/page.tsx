"use client";

import { useEffect, useState } from "react";
import { storeSsoAccessToken } from "@/lib/api";

/**
 * Thin SSO callback landing page.
 * When the IdP redirects back through a proxy that exposes tokens as query
 * params (or when testing locally), persist the access token for subsequent
 * API calls. Production callback normally returns JSON from the hosted API.
 */
export default function SsoCallbackPage() {
  const [status, setStatus] = useState("Completing SSO sign-in…");

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("access_token");
    if (token) {
      storeSsoAccessToken(token);
      setStatus("Signed in. You can close this window or return to the dashboard.");
      return;
    }
    setStatus(
      "SSO callback reached. If your IdP completed successfully, tokens were issued by /api/v1/auth/sso/callback.",
    );
  }, []);

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-950 px-6 text-slate-100">
      <p className="font-mono text-sm text-slate-300">{status}</p>
    </main>
  );
}
