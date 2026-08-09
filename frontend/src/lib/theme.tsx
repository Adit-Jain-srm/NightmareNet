"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  useCallback,
  type ReactNode,
} from "react";

type Theme = "dark" | "light" | "system";
type ResolvedTheme = "dark" | "light";

interface ThemeContextValue {
  theme: Theme;
  resolvedTheme: ResolvedTheme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

const STORAGE_KEY = "nightmarenet-theme";

function getSystemTheme(): ResolvedTheme {
  if (typeof window === "undefined") return "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("system");
  const [systemTheme, setSystemTheme] = useState<ResolvedTheme>("dark");
  const [mounted, setMounted] = useState(false);

  const resolvedTheme = useMemo<ResolvedTheme>(
    () => (theme === "system" ? systemTheme : theme),
    [theme, systemTheme]
  );

  useEffect(() => {
    setMounted(true);
    const stored = localStorage.getItem(STORAGE_KEY) as Theme | null;
    if (stored && ["dark", "light", "system"].includes(stored)) {
      setThemeState(stored);
    }
    setSystemTheme(getSystemTheme());
    const root = document.documentElement;
    if (!root.classList.contains("dark") && !root.classList.contains("light")) {
      root.classList.add("dark");
    }
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const root = document.documentElement;
    root.classList.remove("dark", "light");
    root.classList.add(resolvedTheme);

    if (resolvedTheme === "light") {
      root.style.setProperty("--color-void", "#ffffff");
      root.style.setProperty("--color-abyss", "#f8fafc");
      root.style.setProperty("--color-deep", "#e2e8f0");
      root.style.setProperty("--color-surface", "#cbd5e1");
      root.style.setProperty("--color-muted", "#475569");
      root.style.setProperty("--color-text", "#0f172a");
      root.style.setProperty("--color-text-dim", "#334155");
    } else {
      root.style.setProperty("--color-void", "#030712");
      root.style.setProperty("--color-abyss", "#0a0f1e");
      root.style.setProperty("--color-deep", "#141b2d");
      root.style.setProperty("--color-surface", "#1e293b");
      root.style.setProperty("--color-muted", "#64748b");
      root.style.setProperty("--color-text", "#f1f5f9");
      root.style.setProperty("--color-text-dim", "#94a3b8");
    }
  }, [resolvedTheme, mounted]);

  useEffect(() => {
    if (!mounted || theme !== "system") return;

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = () => {
      setSystemTheme(getSystemTheme());
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [theme, mounted]);

  const setTheme = useCallback((newTheme: Theme) => {
    setThemeState(newTheme);
    localStorage.setItem(STORAGE_KEY, newTheme);
  }, []);

  const toggleTheme = useCallback(() => {
    const next = resolvedTheme === "dark" ? "light" : "dark";
    setTheme(next);
  }, [resolvedTheme, setTheme]);

  if (!mounted) {
    return <>{children}</>;
  }

  return (
    <ThemeContext.Provider
      value={{ theme, resolvedTheme, setTheme, toggleTheme }}
    >
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    return {
      theme: "system" as Theme,
      resolvedTheme: "dark" as ResolvedTheme,
      setTheme: () => {},
      toggleTheme: () => {},
    };
  }
  return context;
}
