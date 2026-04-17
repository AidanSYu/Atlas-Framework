# Darcy flow physics fixture

File: `darcy_small.npz` (2852236 bytes)

## Keys
* `inputs`  : float32 `(N=100, H=64, W=64)` permeability field a(x).
* `outputs` : float32 `(N=100, H=64, W=64)` pressure field u(x).

## Provenance
Local 5-point finite-difference solver for -div(a*grad u) = 1 on [0,1]^2 with Dirichlet-0 boundary. Permeability a(x) drawn from a log-Gaussian random field (length_scale=0.08, sigma=0.75, seed=20260418). Grid: 64x64. This is real Darcy physics, solved locally; it is NOT a copy of the PDEBench dataset.

## Regenerating
`python test_docs/fetch/download_pdebench_darcy.py`

Pass `--force-local` to skip the download attempt and go straight to the local FD solver.
