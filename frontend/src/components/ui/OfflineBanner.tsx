"use client";

import { useEffect, useState } from "react";

export function OfflineBanner() {
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    const updateNetworkState = () => {
      setIsOnline(navigator.onLine);
    };

    updateNetworkState();

    window.addEventListener("online", updateNetworkState);
    window.addEventListener("offline", updateNetworkState);

    return () => {
      window.removeEventListener("online", updateNetworkState);
      window.removeEventListener("offline", updateNetworkState);
    };
  }, []);

  if (isOnline) {
    return null;
  }

  return (
    <div
      role="status"
      aria-live="polite"
      className="sticky top-0 z-[100] border-b border-amber-300/40 bg-amber-950 px-4 py-3 text-center text-sm font-medium text-amber-100 shadow-lg"
    >
      You are offline. Live metrics and API-backed actions may be unavailable
      until your connection returns.
    </div>
  );
}

export default OfflineBanner;
