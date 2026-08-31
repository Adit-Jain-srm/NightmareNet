/**
 * Lightweight Sentry-compatible error reporting.
 *
 * When NEXT_PUBLIC_SENTRY_DSN is unset, all calls are no-ops (no SDK bundled).
 * In development, errors are logged to the console only.
 */

const MAX_REPORTS_PER_MINUTE = 10;
const SESSION_KEY = "nn-error-session";

let sessionId: string | null = null;
let reportTimes: number[] = [];
let initialized = false;

export type ErrorReportContext = {
  componentStack?: string;
  source?: string;
};

type ParsedDsn = {
  publicKey: string;
  host: string;
  projectId: string;
};

function getDsn(): string {
  return process.env.NEXT_PUBLIC_SENTRY_DSN?.trim() ?? "";
}

function isProduction(): boolean {
  return process.env.NODE_ENV === "production";
}

function getSessionId(): string {
  if (typeof window === "undefined") {
    return "server";
  }
  if (sessionId) {
    return sessionId;
  }
  try {
    const existing = sessionStorage.getItem(SESSION_KEY);
    if (existing) {
      sessionId = existing;
      return existing;
    }
    const id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `sess-${Date.now()}`;
    sessionStorage.setItem(SESSION_KEY, id);
    sessionId = id;
    return id;
  } catch {
    return "anonymous";
  }
}

function getRoute(): string {
  if (typeof window === "undefined") {
    return "";
  }
  return `${window.location.pathname}${window.location.search}`;
}

function parseDsn(dsn: string): ParsedDsn | null {
  const match = dsn.match(/^https?:\/\/([^@]+)@([^/]+)\/(.+)$/);
  if (!match) {
    return null;
  }
  return {
    publicKey: match[1],
    host: match[2],
    projectId: match[3],
  };
}

function withinRateLimit(): boolean {
  const now = Date.now();
  reportTimes = reportTimes.filter((t) => now - t < 60_000);
  if (reportTimes.length >= MAX_REPORTS_PER_MINUTE) {
    return false;
  }
  reportTimes.push(now);
  return true;
}

async function sendToSentry(error: Error, context?: ErrorReportContext): Promise<void> {
  const dsn = getDsn();
  const parsed = parseDsn(dsn);
  if (!parsed || !withinRateLimit()) {
    return;
  }

  const eventId =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().replace(/-/g, "")
      : String(Date.now());

  const event = {
    event_id: eventId,
    timestamp: new Date().toISOString(),
    platform: "javascript",
    level: "error",
    logger: "nightmarenet.frontend",
    message: { formatted: error.message },
    exception: {
      values: [
        {
          type: error.name,
          value: error.message,
          stacktrace: error.stack
            ? {
                frames: error.stack.split("\n").slice(1, 6).map((line) => ({
                  filename: line.trim(),
                })),
              }
            : undefined,
        },
      ],
    },
    tags: {
      source: context?.source ?? "unknown",
    },
    extra: {
      route: getRoute(),
      session_id: getSessionId(),
      component_stack: context?.componentStack,
    },
  };

  const url = `https://${parsed.host}/api/${parsed.projectId}/store/`;
  const auth = `Sentry sentry_version=7, sentry_key=${parsed.publicKey}, sentry_client=nightmarenet-frontend/1.0`;

  try {
    await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Sentry-Auth": auth,
      },
      body: JSON.stringify(event),
      keepalive: true,
    });
  } catch {
    // Reporting must never throw back to callers.
  }
}

export function reportError(error: unknown, context?: ErrorReportContext): void {
  const err = error instanceof Error ? error : new Error(String(error));

  if (!isProduction()) {
    if (process.env.NODE_ENV !== "test") {
      console.error("[NightmareNet]", err.message, context ?? {});
    }
    return;
  }

  if (!getDsn()) {
    return;
  }

  void sendToSentry(err, context);
}

export function initErrorReporting(): void {
  if (initialized || typeof window === "undefined") {
    return;
  }
  initialized = true;

  window.addEventListener("unhandledrejection", (event) => {
    reportError(event.reason ?? event, { source: "unhandledrejection" });
  });
}
