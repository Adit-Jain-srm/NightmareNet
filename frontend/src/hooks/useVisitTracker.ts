import { useState, useEffect, useCallback, useMemo } from "react";
import { getSidebarPrefs, saveSidebarPrefs, clearSidebarPrefs, SidebarPrefs } from "../lib/storage";

export function useVisitTracker<T extends string>(validKeys: readonly T[]) {
  const [prefs, setPrefs] = useState<SidebarPrefs>({
    version: 1,
    visitCounts: {},
    customOrder: [],
  });
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setPrefs(getSidebarPrefs());
    setIsLoaded(true);
  }, []);

  const totalVisits = useMemo(() => {
    return Object.values(prefs.visitCounts).reduce((acc, curr) => acc + curr, 0);
  }, [prefs.visitCounts]);

  const registerVisit = useCallback(
    (key: T) => {
      if (!(validKeys as readonly string[]).includes(key)) return;
      
      setPrefs((prev) => {
        const newCounts = { ...prev.visitCounts };
        newCounts[key] = (newCounts[key] || 0) + 1;
        const next = { ...prev, visitCounts: newCounts };
        saveSidebarPrefs(next);
        return next;
      });
    },
    [validKeys]
  );

  const setCustomOrder = useCallback((keys: T[]) => {
    setPrefs((prev) => {
      const next = { ...prev, customOrder: keys };
      saveSidebarPrefs(next);
      return next;
    });
  }, []);

  const reset = useCallback(() => {
    clearSidebarPrefs();
    setPrefs({
      version: 1,
      visitCounts: {},
      customOrder: [],
    });
  }, []);

  useEffect(() => {
    const handleReset = () => reset();
    window.addEventListener("reset-sidebar-prefs", handleReset);
    return () => window.removeEventListener("reset-sidebar-prefs", handleReset);
  }, [reset]);

  const customOrder = useMemo(() => {
    return prefs.customOrder.filter((k): k is T => (validKeys as readonly string[]).includes(k));
  }, [prefs.customOrder, validKeys]);

  const visitCounts = useMemo(() => {
    const counts = {} as Record<T, number>;
    for (const [k, v] of Object.entries(prefs.visitCounts)) {
      if ((validKeys as readonly string[]).includes(k)) {
        counts[k as T] = v;
      }
    }
    return counts;
  }, [prefs.visitCounts, validKeys]);

  return {
    isLoaded,
    totalVisits,
    visitCounts,
    customOrder,
    registerVisit,
    setCustomOrder,
    reset,
  };
}
