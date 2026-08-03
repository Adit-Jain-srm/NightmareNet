import type { Metadata, Viewport } from "next";
import Script from "next/script";
import { ThemeProvider } from "@/lib/theme";
import SkipLink from "@/components/a11y/SkipLink";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { OfflineBanner } from "@/components/ui/OfflineBanner";
import "./globals.css";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#030712",
};

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ||
      "https://frontend-aj5.vercel.app"
  ),

  title: "NightmareNet — Autonomous AI Self-Improvement Platform",

  description:
    "Force neural networks to learn invariant structures through Dream & Nightmare cycles. Autonomous training, adversarial stress-testing, and knowledge compression.",

  manifest: "/manifest.json",

  keywords: [
    "AI",
    "machine learning",
    "neural networks",
    "adversarial training",
    "model compression",
    "dream",
    "nightmare",
    "robustness",
  ],

  authors: [{ name: "Adit Jain" }],

  openGraph: {
    title: "NightmareNet — Autonomous AI Self-Improvement",
    description:
      "Dream & Nightmare cycles that force models to learn what matters.",
    url: "/",
    siteName: "NightmareNet",
    locale: "en_US",
    type: "website",
  },

  twitter: {
    card: "summary_large_image",
    title: "NightmareNet — Autonomous AI Self-Improvement",
    description:
      "Dream & Nightmare cycles that force models to learn what matters.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className="h-full antialiased"
      suppressHydrationWarning
    >
      <body
        className="scanlines min-h-full flex flex-col bg-void text-text font-sans"
        suppressHydrationWarning
      >
      <OfflineBanner />

  <ErrorBoundary
    fallbackTitle="NightmareNet encountered an unexpected error"
    fallbackMessage="The application could not continue. Retry the page content or report the problem."
  >
        <Script
          id="theme-initializer"
          strategy="beforeInteractive"
        >
          {`
            (function () {
              try {
                var theme = localStorage.getItem("nightmarenet-theme");
                var root = document.documentElement;

                if (theme === "light") {
                  root.classList.add("light");
                  root.classList.remove("dark");
                } else {
                  root.classList.add("dark");
                  root.classList.remove("light");
                }
              } catch (error) {
                document.documentElement.classList.add("dark");
              }
            })();
          `}
        </Script>

        <SkipLink />

        <ThemeProvider>
          {children}
        </ThemeProvider>
        </ErrorBoundary>
      </body>
    </html>
  );
}