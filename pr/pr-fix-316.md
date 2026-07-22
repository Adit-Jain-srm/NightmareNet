## Description
This PR resolves issue #316 by replacing the hardcoded "What's New" bullet points with a dynamic live changelog feed.

## Changes Made
- Created a new Next.js API route `frontend/src/app/api/changelog/route.ts` that parses `CHANGELOG.md` at runtime and extracts the latest 5 entries with their respective dates, versions, and optional PR links.
- Updated `frontend/src/components/dashboard/WhatsNew.tsx` to fetch data from `/api/changelog`.
- Refactored the UI of the `WhatsNew` card to display the entries correctly with date, version tags, and an optional link to the GitHub PR.
- Added graceful fallback handling: if the fetch fails or the changelog is empty, the card displays "Check GitHub for the latest updates!".
- Removed unused local constants (such as `BULLETS`) and associated icon imports from `WhatsNew.tsx`.

## Acceptance Criteria Met
- [x] WhatsNew card displays real entries from CHANGELOG.md
- [x] At least the 5 most recent entries shown
- [x] Each entry has a date and optional link to PR/release
- [x] Hardcoded bullets removed
- [x] Card handles empty/missing CHANGELOG gracefully (fallback text)
