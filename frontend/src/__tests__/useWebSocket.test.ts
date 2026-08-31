import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { MAX_RECONNECT_ATTEMPTS } from "@/lib/websocket";
import { useWebSocket } from "@/hooks/useWebSocket";

vi.mock("@/lib/websocket", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/websocket")>();
  return {
    ...actual,
    // Fixed delay so reconnect tests stay deterministic under fake timers.
    nextBackoffMs: () => 1000,
  };
});

type Handler = ((event?: Event | MessageEvent | CloseEvent) => void) | null;

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static OPEN = 1;
  static CLOSED = 3;

  url: string;
  readyState = 0;
  onopen: Handler = null;
  onmessage: Handler = null;
  onerror: Handler = null;
  onclose: Handler = null;
  closeCalls = 0;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  close() {
    this.closeCalls += 1;
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.(new Event("close") as CloseEvent);
  }

  open() {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.(new Event("open"));
  }

  emitMessage(data: unknown) {
    this.onmessage?.(
      new MessageEvent("message", { data: JSON.stringify(data) }),
    );
  }

  emitError() {
    this.onerror?.(new Event("error"));
  }

  emitClose() {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.(new Event("close") as CloseEvent);
  }
}

const WS_URL = "ws://localhost/ws/runs/test-run";

function lastSocket(): MockWebSocket {
  const ws = MockWebSocket.instances.at(-1);
  if (!ws) throw new Error("expected a WebSocket instance");
  return ws;
}

function flushConnect() {
  act(() => {
    vi.advanceTimersByTime(0);
  });
}

describe("useWebSocket", () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    vi.useFakeTimers();
    vi.stubGlobal("WebSocket", MockWebSocket);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("opens a WebSocket with the given URL on mount", () => {
    renderHook(() => useWebSocket({ url: WS_URL }));
    flushConnect();

    expect(MockWebSocket.instances).toHaveLength(1);
    expect(lastSocket().url).toBe(WS_URL);
  });

  it("moves status to connected when the socket opens", () => {
    const { result } = renderHook(() => useWebSocket({ url: WS_URL }));
    flushConnect();

    act(() => {
      lastSocket().open();
    });

    expect(result.current.status).toBe("connected");
    expect(result.current.attempt).toBe(0);
  });

  it("reconnects with backoff after an unexpected close", () => {
    const { result } = renderHook(() => useWebSocket({ url: WS_URL }));
    flushConnect();

    act(() => {
      lastSocket().open();
    });
    expect(result.current.status).toBe("connected");

    act(() => {
      lastSocket().emitClose();
    });
    expect(result.current.status).toBe("reconnecting");
    expect(result.current.attempt).toBe(1);

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(MockWebSocket.instances).toHaveLength(2);

    act(() => {
      lastSocket().open();
    });
    expect(result.current.status).toBe("connected");
    expect(result.current.attempt).toBe(0);
  });

  it("delivers parsed JSON messages to onMessage", () => {
    const onMessage = vi.fn();
    renderHook(() => useWebSocket({ url: WS_URL, onMessage }));
    flushConnect();

    act(() => {
      lastSocket().open();
      lastSocket().emitMessage({ type: "phase", phase: "dream" });
    });

    expect(onMessage).toHaveBeenCalledWith({ type: "phase", phase: "dream" });
  });

  it("closes the socket and clears timers on unmount", () => {
    const { unmount } = renderHook(() => useWebSocket({ url: WS_URL }));
    flushConnect();

    const ws = lastSocket();
    act(() => {
      ws.open();
    });

    unmount();

    expect(ws.closeCalls).toBeGreaterThanOrEqual(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("updates status through connecting and disconnected paths", () => {
    const { result } = renderHook(() => useWebSocket({ url: WS_URL }));
    expect(result.current.status).toBe("disconnected");

    flushConnect();
    act(() => {
      lastSocket().open();
    });
    expect(result.current.status).toBe("connected");

    act(() => {
      result.current.disconnect();
    });
    expect(result.current.status).toBe("disconnected");
    expect(result.current.attempt).toBe(0);
  });

  it("closes the socket on error so reconnection can run", () => {
    const { result } = renderHook(() => useWebSocket({ url: WS_URL }));
    flushConnect();

    act(() => {
      lastSocket().open();
    });

    act(() => {
      lastSocket().emitError();
    });

    // onerror -> close() -> onclose -> scheduleRetry
    expect(result.current.status).toBe("reconnecting");
    expect(lastSocket().closeCalls).toBeGreaterThanOrEqual(1);
  });

  it("stops reconnecting after MAX_RECONNECT_ATTEMPTS", () => {
    const { result } = renderHook(() => useWebSocket({ url: WS_URL }));
    flushConnect();

    act(() => {
      lastSocket().open();
    });

    for (let i = 0; i < MAX_RECONNECT_ATTEMPTS; i++) {
      act(() => {
        lastSocket().emitClose();
      });
      act(() => {
        vi.advanceTimersByTime(1000);
      });
    }

    // One more close should hit the attempt cap and stay disconnected.
    act(() => {
      lastSocket().emitClose();
    });

    expect(result.current.status).toBe("disconnected");
    expect(result.current.attempt).toBe(MAX_RECONNECT_ATTEMPTS);
  });

  it("does not connect when enabled is false or url is null", () => {
    const { rerender } = renderHook(
      ({ url, enabled }: { url: string | null; enabled: boolean }) =>
        useWebSocket({ url, enabled }),
      { initialProps: { url: null as string | null, enabled: true } },
    );
    flushConnect();
    expect(MockWebSocket.instances).toHaveLength(0);

    rerender({ url: WS_URL, enabled: false });
    flushConnect();
    expect(MockWebSocket.instances).toHaveLength(0);
  });

  it("fires onReconnect after a successful reconnect", () => {
    const onReconnect = vi.fn();
    renderHook(() => useWebSocket({ url: WS_URL, onReconnect }));
    flushConnect();

    act(() => {
      lastSocket().open();
    });
    expect(onReconnect).not.toHaveBeenCalled();

    act(() => {
      lastSocket().emitClose();
    });
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    act(() => {
      lastSocket().open();
    });

    expect(onReconnect).toHaveBeenCalledTimes(1);
  });
});
