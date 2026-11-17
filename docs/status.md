# Status

Last updated 2026-11-16. Earlier day-by-day status entries are in this file's git history; decisions are in
[decisions.md](decisions.md) and every paid session is listed in reports/SESSIONS.md.

## Flat-ground tracking on LAFAN1 (step 0): done

| Line | Variant | Result (LAFAN40 whole clip, weight average, seed 0) | Decision |
|---|---|---|---|
| Reproduction | `official-contract` | 40/40, 0.818; noise 40/40, 0.807; CPU MuJoCo 40/40, 0.818; BONES40 held-out 36/40 | D040 |
| Opt 1 | `sim2real-dr` | 39/40, 0.793; 2-step delay 38/40 (reproduction 0/40) | D042 |
| Opt 2 | `sampler-norm` + `wm-hard-mining` | 40/40, 0.781; fall-and-get-up 0.721 | D044, D046 |
| Opt 3 | `optim` (Muon on hidden layers) | 40/40, 0.827 | D049 |

Findings and reasons: [NOTES.md](NOTES.md). Pipeline rules: [PIPELINE.md](PIPELINE.md).

## Stairs, terrain level 1 (step 1): done

Variant `terrain-l1`, warm-started from Opt 3. 20 cm steps, 10 s windows: no fall 0.93 up / 0.92 down / 1.00 random,
climbed 0.90 / 0.79 / 0.98; 0.00–0.03 with the scan flattened; flat LAFAN40 40/40, 0.824. One seed. D047–D050,
[NOTES §10](NOTES.md#10-stairs-terrain-level-1).

## Open items for step 0

- Second seed for every line.
- Muon versus weight decay alone (D049 caveat).
- Remaining world-model-side differences from the official code (`not_aligned` in `variants/cuboi-g1/official-contract.json`).
- BONES-SEED training (out of scope this round, PIPELINE §12).
