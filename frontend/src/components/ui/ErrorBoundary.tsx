"use client";

import Link from "next/link";
import {
  Component,
  type ErrorInfo,
  type ReactNode,
} from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallbackTitle?: string;
  fallbackMessage?: string;
  reportHref?: string;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = {
    error: null,
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    if (process.env.NODE_ENV !== "test") {
      console.error("NightmareNet UI error:", error, errorInfo);
    }

    this.props.onError?.(error, errorInfo);
  }

  private handleRetry = (): void => {
    this.setState({ error: null });
    this.props.onReset?.();
  };

  render(): ReactNode {
    const { error } = this.state;

    if (!error) {
      return this.props.children;
    }

    const {
      fallbackTitle = "This section could not be loaded",
      fallbackMessage = "An unexpected error occurred. You can retry this section without refreshing the whole application.",
      reportHref = "https://github.com/Adit-Jain-srm/NightmareNet/issues/new/choose",
    } = this.props;

    return (
      <section
        role="alert"
        aria-live="assertive"
        className="rounded-2xl border border-red-400/30 bg-red-950/20 p-6 text-left shadow-lg"
      >
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-red-300">
          Recovery mode
        </p>

        <h2 className="mt-2 text-xl font-semibold text-white">
          {fallbackTitle}
        </h2>

        <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
          {fallbackMessage}
        </p>

        {process.env.NODE_ENV === "development" && (
          <details className="mt-4 rounded-lg border border-white/10 bg-black/20 p-3">
            <summary className="cursor-pointer text-sm font-medium text-slate-200">
              Technical details
            </summary>
            <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-xs text-red-200">
              {error.message}
            </pre>
          </details>
        )}

        <div className="mt-5 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={this.handleRetry}
            className="rounded-lg bg-white px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-slate-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
          >
            Retry
          </button>

          <Link
            href={reportHref}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-lg border border-white/20 px-4 py-2 text-sm font-semibold text-white transition hover:bg-white/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
          >
            Report this issue
          </Link>
        </div>
      </section>
    );
  }
}

export default ErrorBoundary;
