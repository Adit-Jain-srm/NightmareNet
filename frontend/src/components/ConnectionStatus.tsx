"use client";

import { Wifi, WifiOff, RefreshCw } from "lucide-react";
import type { WsConnectionStatus } from "@/lib/websocket";
import { MAX_RECONNECT_ATTEMPTS } from "@/lib/websocket";

interface ConnectionStatusProps {
  status: WsConnectionStatus;
  attempt?: number;
  onReconnect?: () => void;
}

export function ConnectionStatus({
  status,
  attempt = 0,
  onReconnect,
}: ConnectionStatusProps) {
  if (status === "connected") {
    return (
      <div className="inline-flex items-center gap-1.5 text-xs text-emerald-400/80">
        <Wifi className="w-3.5 h-3.5" />
        Live
      </div>
    );
  }

  if (status === "reconnecting") {
    return (
      <div className="inline-flex items-center gap-1.5 text-xs text-amber-300/90">
        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
        Reconnecting{attempt > 0 ? ` (${attempt}/${MAX_RECONNECT_ATTEMPTS})` : "…"}
      </div>
    );
  }

  return (
    <div className="inline-flex items-center gap-2 text-xs text-red-300/90">
      <WifiOff className="w-3.5 h-3.5" />
      <span>Disconnected</span>
      {onReconnect && (
        <button
          type="button"
          onClick={onReconnect}
          className="px-2 py-0.5 rounded border border-white/15 hover:bg-white/5 text-white cursor-pointer"
        >
          Reconnect
        </button>
      )}
    </div>
  );
}
