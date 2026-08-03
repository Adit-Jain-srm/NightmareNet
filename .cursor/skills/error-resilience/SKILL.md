---
name: error-resilience
description: >-
  Build resilient systems with proper error handling, retry patterns, circuit
  breakers, graceful degradation, and timeout management. Use when implementing
  API calls, database connections, external service integrations, background jobs,
  or any code that can fail. Also use when the user mentions retry logic, circuit
  breaker, graceful degradation, error boundary, timeout handling, dead letter queue,
  or asks how to make code more resilient.
---

# Error Resilience Patterns

## Overview

Production code fails. Networks drop, services crash, databases timeout. This skill teaches patterns that keep your system running when dependencies don't.

## When to Use

- Any HTTP/API call to an external service
- Database operations that can timeout
- Background job processing
- Event/message consumers
- Any I/O operation in production code
- User asked "how do I handle errors properly"
- User asked "how do I retry failed operations"

## Quick Start: The Minimum Viable Resilience

Every external call needs AT MINIMUM:

```typescript
async function resilientCall<T>(
  fn: () => Promise<T>,
  options: { retries?: number; timeout?: number; fallback?: T } = {}
): Promise<T> {
  const { retries = 3, timeout = 5000, fallback } = options;
  
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeout);
      const result = await fn();
      clearTimeout(timer);
      return result;
    } catch (error) {
      if (attempt === retries) {
        if (fallback !== undefined) return fallback;
        throw error;
      }
      await sleep(exponentialBackoff(attempt));
    }
  }
  throw new Error('Unreachable');
}

function exponentialBackoff(attempt: number): number {
  const base = 1000;
  const jitter = Math.random() * 500;
  return Math.min(base * Math.pow(2, attempt - 1) + jitter, 30000);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
```

## Pattern 1: Retry with Exponential Backoff

**When:** Transient failures (network blips, rate limits, temporary unavailability)

```typescript
// Configuration
const RETRY_CONFIG = {
  maxAttempts: 3,
  baseDelay: 1000,      // 1s, 2s, 4s, 8s...
  maxDelay: 30000,      // Never wait > 30s
  retryableErrors: [
    'ECONNRESET', 'ETIMEDOUT', 'ECONNREFUSED',
    'EPIPE', 'EAI_AGAIN', 'EHOSTUNREACH'
  ],
  retryableStatuses: [408, 429, 500, 502, 503, 504]
};

async function withRetry<T>(
  fn: () => Promise<T>,
  config = RETRY_CONFIG
): Promise<T> {
  let lastError: Error;
  
  for (let attempt = 1; attempt <= config.maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;
      
      if (!isRetryable(error, config)) throw error;
      if (attempt === config.maxAttempts) throw error;
      
      const delay = Math.min(
        config.baseDelay * Math.pow(2, attempt - 1) + Math.random() * 500,
        config.maxDelay
      );
      
      console.warn(`Attempt ${attempt} failed, retrying in ${delay}ms:`, error.message);
      await sleep(delay);
    }
  }
  throw lastError!;
}

function isRetryable(error: any, config: typeof RETRY_CONFIG): boolean {
  if (error.code && config.retryableErrors.includes(error.code)) return true;
  if (error.status && config.retryableStatuses.includes(error.status)) return true;
  if (error.message?.includes('rate limit')) return true;
  return false;
}
```

**Never retry:** 400, 401, 403, 404, 422 (client errors = your bug, retrying won't fix it)

## Pattern 2: Circuit Breaker

**When:** Downstream service is DOWN (not just slow). Stop hammering it.

```typescript
class CircuitBreaker {
  private failures = 0;
  private lastFailure = 0;
  private state: 'closed' | 'open' | 'half-open' = 'closed';
  
  constructor(
    private threshold = 5,       // Open after 5 failures
    private resetTimeout = 30000 // Try again after 30s
  ) {}
  
  async call<T>(fn: () => Promise<T>, fallback?: () => T): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() - this.lastFailure > this.resetTimeout) {
        this.state = 'half-open';
      } else {
        if (fallback) return fallback();
        throw new Error('Circuit breaker is OPEN — service unavailable');
      }
    }
    
    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      if (fallback) return fallback();
      throw error;
    }
  }
  
  private onSuccess() {
    this.failures = 0;
    this.state = 'closed';
  }
  
  private onFailure() {
    this.failures++;
    this.lastFailure = Date.now();
    if (this.failures >= this.threshold) {
      this.state = 'open';
      console.error(`Circuit OPEN after ${this.failures} failures`);
    }
  }
}

// Usage:
const paymentCircuit = new CircuitBreaker(3, 60000);
const result = await paymentCircuit.call(
  () => stripe.charges.create(params),
  () => ({ status: 'queued', message: 'Payment queued for retry' })
);
```

## Pattern 3: Timeout + Abort

**When:** Operations that can hang forever (DNS, DB connections, external APIs)

```typescript
async function withTimeout<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  ms: number,
  errorMessage = `Operation timed out after ${ms}ms`
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  
  try {
    const result = await fn(controller.signal);
    return result;
  } catch (error: any) {
    if (error.name === 'AbortError') {
      throw new Error(errorMessage);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

// Usage with fetch:
const data = await withTimeout(
  (signal) => fetch('https://api.example.com/data', { signal }).then(r => r.json()),
  5000,
  'API did not respond within 5 seconds'
);
```

## Pattern 4: Graceful Degradation

**When:** Feature is nice-to-have, core functionality must continue without it

```typescript
async function withGracefulDegradation<T>(
  primary: () => Promise<T>,
  fallback: T | (() => T | Promise<T>),
  options: { log?: boolean; metric?: string } = {}
): Promise<T> {
  try {
    return await primary();
  } catch (error) {
    if (options.log !== false) {
      console.warn(`Degraded: ${options.metric || 'unknown'}`, error);
    }
    return typeof fallback === 'function' ? (fallback as Function)() : fallback;
  }
}

// Examples:
const recommendations = await withGracefulDegradation(
  () => mlService.getRecommendations(userId),
  [],  // Show no recommendations rather than error page
  { metric: 'recommendations.degraded' }
);

const userAvatar = await withGracefulDegradation(
  () => avatarService.get(userId),
  '/default-avatar.png',
  { metric: 'avatar.degraded' }
);
```

## Pattern 5: Dead Letter Queue (DLQ)

**When:** Background job processing — failed jobs need investigation, not silent drops

```typescript
interface DLQEntry {
  originalPayload: any;
  error: string;
  failedAt: string;
  attempts: number;
  queue: string;
}

class JobProcessor {
  private maxAttempts = 3;
  
  async process(job: any, handler: (job: any) => Promise<void>): Promise<void> {
    for (let attempt = 1; attempt <= this.maxAttempts; attempt++) {
      try {
        await handler(job);
        return; // Success
      } catch (error: any) {
        if (attempt === this.maxAttempts) {
          await this.sendToDLQ({
            originalPayload: job,
            error: error.message,
            failedAt: new Date().toISOString(),
            attempts: attempt,
            queue: 'main'
          });
          return; // Don't throw — job is safely in DLQ
        }
        await sleep(exponentialBackoff(attempt));
      }
    }
  }
  
  private async sendToDLQ(entry: DLQEntry): Promise<void> {
    // Store in DLQ table/queue for manual investigation
    console.error('Job moved to DLQ:', entry);
    // await db.insert('dead_letter_queue', entry);
  }
}
```

## Decision Matrix

| Situation | Pattern | Why |
|-----------|---------|-----|
| API call might fail temporarily | Retry + Backoff | Transient errors resolve on retry |
| Service is completely down | Circuit Breaker | Stop wasting resources on dead service |
| Operation might hang forever | Timeout + Abort | Prevent thread/connection exhaustion |
| Feature failure shouldn't crash app | Graceful Degradation | Core UX survives |
| Background job fails repeatedly | Dead Letter Queue | Don't lose data, investigate later |
| Multiple services, any might fail | Combine all above | Defense in depth |

## Common Mistakes

- **Retrying non-retryable errors** (400/401/404 — your bug, not theirs)
- **No jitter in backoff** (all clients retry at same time = thundering herd)
- **Infinite retries** (always set maxAttempts — infinite = memory leak + useless)
- **Swallowing errors silently** (catch without logging = invisible failures)
- **Same timeout for all operations** (DNS: 2s, DB query: 5s, file upload: 60s — tune per operation)
- **No circuit breaker for cascading calls** (Service A → B → C, if C is down, A and B pile up)
- **Retrying inside a retry** (nested retries = exponential attempts: 3 × 3 × 3 = 27 attempts)

## Verification

After implementing resilience patterns:
- [ ] Non-retryable errors (400, 401, 404) are NOT retried
- [ ] Retries have exponential backoff WITH jitter
- [ ] Timeouts are set for every external call
- [ ] Circuit breaker prevents cascade when service is down
- [ ] Graceful degradation tested (kill dependency, verify app still works)
- [ ] Failed jobs land in DLQ with full context for debugging
- [ ] Logs show retry attempts and final outcomes
- [ ] Metrics track failure rates per dependency
