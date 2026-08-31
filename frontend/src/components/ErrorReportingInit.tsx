"use client";

import { useEffect } from "react";
import { initErrorReporting } from "@/lib/error-reporting";

/** Registers global client-side error handlers once per page load. */
export function ErrorReportingInit() {
  useEffect(() => {
    initErrorReporting();
  }, []);

  return null;
}
