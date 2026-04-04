import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800", "900"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#030712",
};

export const metadata: Metadata = {
  title: "NightmareNet — Autonomous AI Self-Improvement Platform",
  description:
    "Force neural networks to learn invariant structures through Dream & Nightmare cycles. Autonomous training, adversarial stress-testing, and knowledge compression.",
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
  icons: { icon: "/favicon.ico" },
  openGraph: {
    title: "NightmareNet — Autonomous AI Self-Improvement",
    description:
      "Dream & Nightmare cycles that force models to learn what matters.",
    type: "website",
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
      className={`${inter.variable} ${jetbrainsMono.variable} h-full antialiased`}
    >
      <body className="scanlines min-h-full flex flex-col bg-void text-text font-sans">
        {children}
      </body>
    </html>
  );
}
