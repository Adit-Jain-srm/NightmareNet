/**
 * Storage schema and utilities for NightmareNet frontend.
 */

export interface SidebarPrefsV1 {
  version: 1;
  visitCounts: Record<string, number>;
  customOrder: string[];
}

export type SidebarPrefs = SidebarPrefsV1;

const SIDEBAR_PREFS_KEY = "nightmare_sidebar_prefs";
const CURRENT_VERSION = 1;

function defaultSidebarPrefs(): SidebarPrefs {
  return {
    version: CURRENT_VERSION,
    visitCounts: {},
    customOrder: [],
  };
}

/**
 * Migration logic for future schema updates.
 */
function migrateSidebarPrefs(data: unknown): SidebarPrefs {
  if (!data || typeof data !== "object") {
    return defaultSidebarPrefs();
  }

  const record = data as Record<string, unknown>;

  // Future migrations can be added here
  // if (record.version === 1) {
  //   record = migrateV1ToV2(record);
  // }

  // Fallback if version is unknown
  if (record.version !== CURRENT_VERSION) {
    return defaultSidebarPrefs();
  }

  return record as unknown as SidebarPrefs;
}

export function getSidebarPrefs(): SidebarPrefs {
  if (typeof window === "undefined") {
    return defaultSidebarPrefs();
  }
  
  try {
    const raw = localStorage.getItem(SIDEBAR_PREFS_KEY);
    if (!raw) return defaultSidebarPrefs();
    const parsed = JSON.parse(raw);
    return migrateSidebarPrefs(parsed);
  } catch (err) {
    console.error("Failed to parse sidebar prefs:", err);
    return defaultSidebarPrefs();
  }
}

export function saveSidebarPrefs(prefs: SidebarPrefs): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(SIDEBAR_PREFS_KEY, JSON.stringify(prefs));
  } catch (err) {
    console.error("Failed to save sidebar prefs:", err);
  }
}

export function clearSidebarPrefs(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(SIDEBAR_PREFS_KEY);
}
