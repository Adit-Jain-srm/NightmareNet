/**
 * Retry utility with exponential backoff.
 *
 * Retries on transient errors: network failures, 502, 503, 504, 429.
 * Does NOT retry on client errors: 400, 401, 403, 404, 422.
 * Respects Retry-After header for 429 responses.
 */

/** Status codes that are safe to retry (transient server errors). */
const RETRYABLE_STATUS_CODES = new Set([429, 502, 503, 504]);

/** Status codes that are client errors and should never be retried. */
const NON_RETRYABLE_STATUS_CODES = new Set([400, 401, 403, 404, 422]);

export interface RetryOptions {
  /** Maximum number of retry attempts (default: 3). */
  maxRetries?: number;
  /** Base delay in milliseconds for exponential backoff (default: 1000). */
  baseDelayMs?: number;
  /** Callback fired before each retry — use to show loading state. */
  onRetry?: (attempt: number, delayMs: number) => void;
}

/**
 * Error thrown when a fetch response has a non-retryable HTTP error status.
 * Carries the original Response so callers can inspect status / body.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly response: Response,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Determines whether a fetch failure is safe to retry.
 *
 * - Network errors (TypeError from fetch) → retryable.
 * - ApiError with a retryable status code → retryable.
 * - ApiError with a non-retryable status code → not retryable.
 */
function isRetryable(error: unknown): boolean {
  if (error instanceof TypeError) {
    // Network-level failure (no response received).
    return true;
  }
  if (error instanceof ApiError) {
    return RETRYABLE_STATUS_CODES.has(error.status);
  }
  return false;
}

/**
 * Extracts the retry delay from a 429 Retry-After header, if present.
 * Returns `undefined` when the header is absent or unparseable.
 */
function retryAfterMs(response: Response): number | undefined {
  const header = response.headers.get("Retry-After");
  if (!header) return undefined;

  // Retry-After can be a delta-seconds integer or an HTTP-date string.
  const seconds = parseInt(header, 10);
  if (!isNaN(seconds)) return seconds * 1000;

  const date = Date.parse(header);
  if (!isNaN(date)) return Math.max(0, date - Date.now());

  return undefined;
}

/**
 * Wraps a fetch call with retry logic and exponential backoff.
 *
 * @param fn        An async function that calls `fetch` and returns a Response.
 * @param options   Retry configuration.
 * @returns         The successful Response.
 * @throws          The last error after all retries are exhausted.
 */
export async function withRetry(
  fn: () => Promise<Response>,
  options: RetryOptions = {},
): Promise<Response> {
  const { maxRetries = 3, baseDelayMs = 1000, onRetry } = options;

  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fn();

      // Surface non-OK responses as typed errors so callers can branch on status.
    if (!response.ok) {
  const cloned = response.clone();
  const body = await cloned.json().catch(() => ({})) as Record<string, string>;
  const message = body.detail || body.error || `API error ${response.status}`;

  if (NON_RETRYABLE_STATUS_CODES.has(response.status)) {
    throw new ApiError(message, response.status, response);
  }

  if (RETRYABLE_STATUS_CODES.has(response.status)) {
    lastError = new ApiError(message, response.status, response);

    if (attempt < maxRetries) {
      const delay =
        response.status === 429
          ? (retryAfterMs(response) ?? baseDelayMs * 2 ** attempt)
          : baseDelayMs * 2 ** attempt;

      onRetry?.(attempt + 1, delay);
      await sleep(delay);
      continue;
    }

    throw lastError;
  }

  // Unexpected non-OK status — propagate immediately.
  throw new ApiError(message, response.status, response);
}

      return response;
    } catch (error) {
      // Re-throw non-retryable errors immediately.
      if (!isRetryable(error)) throw error;

      lastError = error;

      if (attempt < maxRetries) {
        const delay = baseDelayMs * 2 ** attempt;
        onRetry?.(attempt + 1, delay);
        await sleep(delay);
      }
    }
  }

  throw lastError;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}