# Compliant open-vantage pano slot

The Machu Picchu fixture (`local_panos/machu*.jpg`, private) is the
**adversarial** fixture: it correctly FAILs `min_content_distance` (stone
walls ~3.1 m from the camera, under the 6 m comfort minimum) and `stereo`
(close-content depth-order). It stays that way on purpose — do NOT relax those
thresholds to make it pass.

This file reserves the slot for a **compliant** fixture: a licensed
open-vantage 360° pano where the nearest real content is comfortably beyond
`min_content_distance_m` (6 m) in every horizon direction, so a run can reach
`shippable: true`. Good candidates: an open field, a plaza, a rooftop, a
mountain overlook — anywhere the camera is not boxed in by near geometry.

## Requirements for a compliant fixture

- Equirectangular 2:1, people-free (or pre-cleaned via the S1 loop).
- Nearest horizon-band content ≥ ~8 m (headroom over the 6 m gate).
- A commercially-cleared license, recorded in the sidecar
  `<name>.jpg.license.json` (schema `license_sidecar`): `source`,
  `license_id`, `scope_note`, `camera_height_m`.
- Place the pano in `local_panos/` (gitignored — source panos are private).
  Only the derived `.sog`/`.ply` and the review page get published.

## When a compliant pano is provided

1. `make run PANO=local_panos/<name>.jpg OUT=runs/<name>` (add
   `--params local_panos/params_<name>.yaml` only if a scale override is
   needed).
2. Confirm the receipt: `min_content_distance` PASS, `stereo` PASS,
   `bg_solid_angle` ≤ 5%, `fidelity_at_origin` reporting, `shippable: true`.
3. Promote it as the accepted baseline: `make accept RUN=runs/<name>`.
4. Deploy its review profile to `docs/` for the Pages viewer.

Until then, the Machu fixture demonstrates the gates firing on an
intentionally non-compliant vantage — which is the point.
