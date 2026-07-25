import type { KeyboardEvent } from "react";

export function activateOnEnterOrSpace(
  event: KeyboardEvent<HTMLElement>,
  action: () => void,
) {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

export function announceCount(
  count: number,
  singular: string,
  plural = `${singular}s`,
) {
  return `${count} ${count === 1 ? singular : plural}`;
}
