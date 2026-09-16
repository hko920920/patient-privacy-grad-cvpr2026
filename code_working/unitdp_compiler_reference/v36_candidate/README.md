# Isolated V3.6 Candidate

This tree contains the provisional external-PLD handler, its three strict
contracts, and its candidate-only tests. It is intentionally outside the
frozen `src/`, `configs/`, and `tests/` trees consumed by the deterministic
V3.5 public-package builder.

The candidate modules extend the existing `unitdp` package only when a G8
script or candidate test appends `v36_candidate/src/unitdp` to
`unitdp.__path__`. The frozen P/F/A implementation files and V3.5 artifacts
remain unchanged.

This is finite evidence for one externally maintained accountant overlay on
the already registered A mechanism and executor. It is not an open-world
plugin guarantee, an independent privacy certificate, or a new mechanism.
