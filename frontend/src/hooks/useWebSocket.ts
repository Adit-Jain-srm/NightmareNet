"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  MAX_RECONNECT_ATTEMPTS,
  nextBackoffMs,
  type WsConnectionStatus,
} from "@/lib/websocket";

export interface UseWebSocketOptions {
  url: string | null;
  enabled?: boolean;
  onMessage?: (data: unknown) => void;
  /** Fired after a successful reconnect so callers can refresh state. */
  onReconnect?: () => void;
}

export interface UseWebSocketResult {
  status: WsConnectionStatus;
  attempt: number;
  reconnect: () => void;
  disconnect: () => void;
}

export function useWebSocket({
  url,
  enabled = true,
  onMessage,
  onReconnect,
}: UseWebSocketOptions): UseWebSocketResult {
  const [status, setStatus] = useState<WsConnectionStatus>("disconnected");
  const [attempt, setAttempt] = useState(0);
  const [connectionKey, setConnectionKey] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const blockedUrlRef = useRef<string | null>(null);
  const hasConnectedRef = useRef(false);
  const previousUrlRef = useRef<string | null>(null);
  const onMessageRef = useRef(onMessage);
  const onReconnectRef = useRef(onReconnect);

  useEffect(() => {
    onMessageRef.current = onMessage;
    onReconnectRef.current = onReconnect;
  }, [onMessage, onReconnect]);

  const disconnect = useCallback(() => {
    blockedUrlRef.current = url;
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setAttempt(0);
    setStatus("disconnected");
    setConnectionKey((key) => key + 1);
  }, [url]);

  const reconnect = useCallback(() => {
    blockedUrlRef.current = null;
    setAttempt(0);
    setStatus("reconnecting");
    setConnectionKey((key) => key + 1);
  }, []);

  useEffect(() => {
    if (!enabled || !url || blockedUrlRef.current === url) return;
    const targetUrl = url;

    if (previousUrlRef.current !== targetUrl) {
      previousUrlRef.current = targetUrl;
      hasConnectedRef.current = false;
    }

    let active = true;
    let retryCount = 0;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    function scheduleRetry() {
      if (!active) return;

      if (retryCount >= MAX_RECONNECT_ATTEMPTS) {
        setStatus("disconnected");
        return;
      }

      const delay = nextBackoffMs(retryCount);
      retryCount += 1;
      setAttempt(retryCount);
      setStatus("reconnecting");
      retryTimer = setTimeout(connect, delay);
    }

    function connect() {
      if (!active) return;

      let ws: WebSocket;
      try {
        ws = new WebSocket(targetUrl);
      } catch {
        scheduleRetry();
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        const reconnected = hasConnectedRef.current;
        hasConnectedRef.current = true;
        retryCount = 0;
        setAttempt(0);
        setStatus("connected");
        if (reconnected) onReconnectRef.current?.();
      };

      ws.onmessage = (event) => {
        try {
          onMessageRef.current?.(JSON.parse(event.data));
        } catch {
          /* ignore bad payloads */
        }
      };

      ws.onerror = () => {
        ws.close();
      };

      ws.onclose = () => {
        if (!active) return;
        wsRef.current = null;
        scheduleRetry();
      };
    }

    retryTimer = setTimeout(connect, 0);

    return () => {
      active = false;
      if (retryTimer) clearTimeout(retryTimer);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [url, enabled, connectionKey]);

  return { status, attempt, reconnect, disconnect };
}
