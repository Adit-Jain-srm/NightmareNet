## Description
This PR resolves issue #315 by wiring the Re-run button in `RunDetail.tsx` to the `POST /api/v1/pipeline/create` endpoint.

## Changes Made
- Added `useRouter` from `next/navigation` to `RunDetail.tsx` to enable redirection.
- Updated `ReRunMenu` to track `loading` state during the API call.
- Replaced the TODO block in the `fire` function with a call to `createPipeline`.
- Included success handling that redirects to the newly created pipeline run detail page (`/run/${res.run_id}`).
- Included error handling that catches any exceptions during the API call and displays an error toast.
- Updated the Re-run `Button` component to display "Re-running..." and use the `disabled` and `loading` states while the API call is in flight.

## Acceptance Criteria Met
- [x] Re-run button calls `/api/v1/pipeline/create` with the run's original config
- [x] Loading indicator shown during API call
- [x] Success redirects to the new run's RunDetail page
- [x] Failure shows descriptive error toast
- [x] TODO comment at L206 removed
- [x] No regression to existing RunDetail rendering
