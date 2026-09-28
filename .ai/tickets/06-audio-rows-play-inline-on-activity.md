# Audio rows play inline on the Activity tab and don't break the Gallery

## What To Build

An `activity_v` row whose `url` ends in `.ogg` or `.wav` renders an
`<audio controls>` element in the Activity table instead of a broken `<img>`;
the Gallery either renders the same element or excludes audio kinds from its
image grid. No new endpoints - the `/images` static mount already serves the
files.

## Blocked By

04.

## Scope

- `frontend/src/tabs/Activity.tsx` `Row`: branch on `item.kind in
  ('music','ambience')` (preferred over sniffing the extension, since `kind`
  is what the ledger says) -> `<audio controls preload="none"
  src={item.url}>`; keep the `<img>` path for everything else. Expanded row
  shows `info` fields that matter for audio: `style`, `bpm`, `duration_s`,
  `loop_start`/`loop_end`, `seam_rms_jump`, `author`, `served_from`.
- Activity filter dropdown gains `music` and `ambience` if it enumerates kinds.
- `frontend/src/tabs/Gallery.tsx`: same branch, or filter audio kinds out of
  the grid with a count line ("3 audio assets - see Activity or Audio").
  Pick the smaller change; audio in the gallery is not a requirement.
- `frontend/src/api.ts`: `ActivityItem` type gains the audio `info` fields
  (optional).
- Rebuild with `scripts/build-frontend.sh`.

## Out Of Scope

The Audio tab (07), loop audition, downloads with a bearer.

## Acceptance Criteria

- [ ] After a `POST /api/audio` (ticket 04), the Activity tab shows the row with a playable audio control; pressing play plays the OGG.
- [ ] A `running` audio row shows the same pending state other kinds do.
- [ ] The Gallery renders without a broken thumbnail for audio rows.
- [ ] Existing image rows are unchanged (spot-check an entity and a tile row).

## Test Seam

Not applicable - presentational; verified in the browser. The data contract
is covered by `verify-audio-api.py` in 04.

## Verification

    bash scripts/build-frontend.sh
    # open http://<host>:8001/#activity, filter kind=music

## Implementation Notes

- `item.url` comes from `generations._url` -> `/images/<basename>`; the mount
  is unauthenticated, so no `fetchObjectUrl` is needed here. Downloads of the
  WAV master through `/api/audio/...` (ticket 07) do need it.
- `preload="none"`: Activity lists many rows; do not fetch every track on
  load.

## Review Focus

That the branch keys on `kind`, not on URL text, and that nothing regresses
for image rows.

## Suggested Route

`/implement`, then `/review-code`.

## Status 2026-09-28

Implemented: `Activity.tsx` renders `<audio controls preload="none">` for
`isAudioKind(kind)` rows (branch on the ledger kind, not the extension) and an
expanded "Audio / Loop / Style chosen by" block from a new `audio` field;
`Gallery.tsx` renders the same control. `generations._attach_audio` fills that
field with one extra query for the audio rows on the page, because
`activity_v` has no `params` and widening a UNION view is a migration. Found
and fixed on the way: `assets.to_url` duplicated the `/images/<basename>`
rule, so Gallery links to audio 404'd once audio moved to `AUDIO_DIR`; it now
delegates to `generations._url`.

Verified: frontend build (tsc + vite) passes; the feed returns `/audio/...`
URLs and loop fields for generated rows and `audio: null` for cache reads;
Gallery asset URLs resolve under `/audio/`. **Not verified: playback in a
browser** - no box above is ticked until someone presses play on the tab.
The Activity filter has no kind selector, so there was nothing to extend.
