import { lazy, type ComponentType } from "react";

export function retryLazy<T extends ComponentType<any>>(
  factory: () => Promise<{ default: T }>,
  retries = 3,
  interval = 1000,
  backoffMultiplier = 2,
): ReturnType<typeof lazy<T>> {
  return lazy(
    () =>
      new Promise<{ default: T }>((resolve, reject) => {
        const attempt = (retriesLeft: number, currentInterval: number) => {
          factory()
            .then(resolve)
            .catch((err: Error) => {
              if (retriesLeft <= 0) {
                reject(err);
                return;
              }
              setTimeout(
                () => attempt(retriesLeft - 1, currentInterval * backoffMultiplier),
                currentInterval,
              );
            });
        };
        attempt(retries, interval);
      }),
  );
}
