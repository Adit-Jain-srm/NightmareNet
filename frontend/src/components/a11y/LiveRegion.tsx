"use client";

type LiveRegionProps = {
  message: string;
  assertive?: boolean;
  atomic?: boolean;
};

export function LiveRegion({ message, assertive = false, atomic = true }: LiveRegionProps) {
  return (
    <div
      className="sr-only"
      role={assertive ? "alert" : "status"}
      aria-live={assertive ? "assertive" : "polite"}
      aria-atomic={atomic}
    >
      {message}
    </div>
  );
}
