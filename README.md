# Enumerative Problems Computation Scripts

Julia and Macaulay2 scripts for computations related to enumerative geometry
problems.

This repository was exported from an Obsidian vault and intentionally contains
only source files for the relevant computations. Directory names from the
original vault are preserved where relevant.

## Contents

- `two_powers_monodromy.jl`
- `ternary_sextic_two_powers_monodromy.jl`
- `ternary_sextic_monodromy_orchestrator.jl`
- `ternary_sextic_email_bundle/`
- `Counting Pentagrams/computation/`
- `Hessian Simplex Map/`
- `ProjectiveGeometry/PR_computations/`

## Projection-ramification scroll scripts

The folder `ProjectiveGeometry/PR_computations/` contains the original
Macaulay2 case files and a newer HomotopyContinuation script:

```bash
julia --project=ProjectiveGeometry/PR_computations -t auto \
  ProjectiveGeometry/PR_computations/scroll_projection_ramification.jl \
  --degrees 2,2 --timeout 300
```

The script uses an affine Grassmannian chart, computes ramification
coefficients in `H^0(P1, E det(E) K_P1)`, and writes resumable/mergeable
checkpoints containing the full basepoint and target state.  The helper
`classify_O22_real_ramification.jl` samples real bidegree `(1,4)` targets in
the `O(2)+O(2)` case and distinguishes real fibers with two real PR preimages
from fibers with a complex-conjugate pair.
