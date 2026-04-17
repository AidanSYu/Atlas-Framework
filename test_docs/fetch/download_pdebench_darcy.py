"""Build the 2D Darcy-flow physics fixture used by the physics_simulator plugin.

Primary path
------------
Download the PDEBench 2D Darcy Flow dataset from the official DaRUS mirror
(https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986),
slice off the first 100 samples, and save them as a compact .npz.

Fallback path
-------------
If the DaRUS endpoint is unreachable (no network, geoblock, dataset moved, etc.)
the script solves the 2D Darcy equation

    -div( a(x) * grad(u(x)) ) = f(x)     x in [0,1]^2
                     u(x) = 0            x on boundary

locally with a 5-point finite-difference discretisation on a 64x64 grid, using
scipy.sparse for the linear solve. The permeability field `a(x)` is drawn from
a log-Gaussian random field so each sample has a non-trivial spatial structure,
matching what PDEBench ships. This is still *real* physics — not random arrays
— the only difference is the source of the permeability fields.

Either path yields an .npz with:
    inputs  : float32 array, shape (100, H, W)   -- permeability field a(x)
    outputs : float32 array, shape (100, H, W)   -- pressure field u(x)

and writes a provenance README next to it.

Usage
-----
    python download_pdebench_darcy.py
    python download_pdebench_darcy.py --force-local   # skip the download attempt
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "physics"
OUTPUT_NPZ = OUTPUT_DIR / "darcy_small.npz"
OUTPUT_README = OUTPUT_DIR / "README.md"

GRID = 64
N_SAMPLES = 100
SEED = 20260418

# PDEBench DaRUS endpoint. The 2D Darcy Flow HDF5 is hosted as a direct file
# on the DaRUS dataset record. The exact file ID may shift as the record is
# updated; we probe a couple of known mirror URLs and fall through if none
# respond.
PDEBENCH_URLS = [
    # Primary: direct DaRUS file access. File IDs are stable per upload.
    # 2D_DarcyFlow_beta1.0_Train.hdf5 at ~350MB - we only pull 100 samples
    "https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/darus-2986/16",
]


# ---------------------------------------------------------------------------
# Fallback: local 2D Darcy finite-difference solver
# ---------------------------------------------------------------------------

def _grf_2d(n: int, rng: np.random.Generator, length_scale: float = 0.1) -> np.ndarray:
    """Draw one Gaussian random field on an n x n grid via spectral synthesis.

    Returns a real-valued field with unit variance and correlation length
    ~length_scale (in fractions of the domain).
    """
    kx = np.fft.fftfreq(n, d=1.0 / n)
    ky = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX ** 2 + KY ** 2
    # Matern-like spectrum
    spectrum = np.exp(-length_scale * np.sqrt(k2))
    spectrum[0, 0] = 0.0
    noise = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    field = np.fft.ifft2(noise * np.sqrt(spectrum)).real
    field -= field.mean()
    std = field.std()
    if std > 0:
        field /= std
    return field.astype(np.float32)


def _build_laplacian_operator(a_interior: np.ndarray, h: float):
    """Assemble the 5-point Darcy operator -div(a grad u) on interior nodes.

    Uses harmonic averaging of `a` on edges, which is the standard finite-volume
    choice for this PDE. `a_interior` has shape (n, n) covering the full grid;
    only interior nodes (1..n-2, 1..n-2) get equations — boundary nodes are
    pinned to Dirichlet 0.
    """
    from scipy.sparse import lil_matrix

    n = a_interior.shape[0]
    m = n - 2  # interior count per axis
    N = m * m
    A = lil_matrix((N, N))

    def idx(i: int, j: int) -> int:
        return (i - 1) * m + (j - 1)

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            k = idx(i, j)
            # Harmonic-average permeability on each face
            a_e = 2.0 * a_interior[i, j] * a_interior[i, j + 1] / (a_interior[i, j] + a_interior[i, j + 1])
            a_w = 2.0 * a_interior[i, j] * a_interior[i, j - 1] / (a_interior[i, j] + a_interior[i, j - 1])
            a_n = 2.0 * a_interior[i, j] * a_interior[i - 1, j] / (a_interior[i, j] + a_interior[i - 1, j])
            a_s = 2.0 * a_interior[i, j] * a_interior[i + 1, j] / (a_interior[i, j] + a_interior[i + 1, j])
            inv_h2 = 1.0 / (h * h)
            A[k, k] = (a_e + a_w + a_n + a_s) * inv_h2
            if j + 1 < n - 1:
                A[k, idx(i, j + 1)] = -a_e * inv_h2
            if j - 1 > 0:
                A[k, idx(i, j - 1)] = -a_w * inv_h2
            if i - 1 > 0:
                A[k, idx(i - 1, j)] = -a_n * inv_h2
            if i + 1 < n - 1:
                A[k, idx(i + 1, j)] = -a_s * inv_h2
    return A.tocsr()


def _solve_darcy_local(n_samples: int, grid: int) -> Tuple[np.ndarray, np.ndarray]:
    """Solve `n_samples` instances of 2D Darcy flow with random permeability."""
    from scipy.sparse.linalg import spsolve

    rng = np.random.default_rng(SEED)
    h = 1.0 / (grid - 1)
    m = grid - 2

    # Uniform forcing f(x) = 1 throughout interior.
    f_interior = np.ones((m * m,), dtype=np.float64)

    inputs = np.zeros((n_samples, grid, grid), dtype=np.float32)
    outputs = np.zeros((n_samples, grid, grid), dtype=np.float32)

    for s in range(n_samples):
        # Permeability: log-Gaussian, a(x) = exp(sigma * field), keeps values > 0
        field = _grf_2d(grid, rng, length_scale=0.08)
        a = np.exp(0.75 * field).astype(np.float64)  # ~ [0.3, 3.0]
        inputs[s] = a.astype(np.float32)

        A = _build_laplacian_operator(a, h)
        u_interior = spsolve(A, f_interior)
        u_full = np.zeros((grid, grid), dtype=np.float64)
        u_full[1:-1, 1:-1] = u_interior.reshape((m, m))
        outputs[s] = u_full.astype(np.float32)

        if (s + 1) % 10 == 0:
            print(f"  solved {s + 1}/{n_samples}")

    return inputs, outputs


# ---------------------------------------------------------------------------
# Primary: attempt PDEBench DaRUS download
# ---------------------------------------------------------------------------

def _try_pdebench(n_samples: int) -> Tuple[np.ndarray, np.ndarray] | None:
    try:
        import urllib.error
        import urllib.request
    except ImportError:
        return None
    try:
        import h5py  # type: ignore
    except ImportError:
        print("  h5py not installed; skipping PDEBench HDF5 path.")
        return None

    for url in PDEBENCH_URLS:
        print(f"  probing {url} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "atlas-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                # Stream a bounded chunk so we don't blow disk on full ~350MB file.
                # But we actually need the full file since HDF5 can't be sliced
                # over HTTP without range requests + a valid HDF5 parser on a
                # partial file. Download to a tempfile only if size looks sane.
                length_header = resp.headers.get("Content-Length")
                if length_header and int(length_header) > 600_000_000:
                    print(f"  file too large ({length_header} bytes); skipping.")
                    continue
                buf = io.BytesIO()
                chunk = resp.read(65536)
                while chunk:
                    buf.write(chunk)
                    if buf.tell() > 600_000_000:
                        print("  streamed past 600MB cap; aborting.")
                        return None
                    chunk = resp.read(65536)
            buf.seek(0)
            with h5py.File(buf, "r") as hf:
                # PDEBench 2D Darcy layout: 'nu' (permeability) + 'tensor' (pressure),
                # both (N, H, W).  Names vary; try common fallbacks.
                key_in = next((k for k in ("nu", "permeability", "a") if k in hf), None)
                key_out = next((k for k in ("tensor", "pressure", "u") if k in hf), None)
                if key_in is None or key_out is None:
                    print(f"  HDF5 keys {list(hf.keys())} don't match expected Darcy schema.")
                    continue
                inp = hf[key_in][:n_samples].astype(np.float32)
                out = hf[key_out][:n_samples].astype(np.float32)
                return inp, out
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            print(f"  download failed: {exc}")
            continue
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-local", action="store_true",
                        help="Skip PDEBench download and go straight to the local FD solver.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs: np.ndarray | None = None
    outputs: np.ndarray | None = None
    provenance = ""

    if not args.force_local:
        print("Attempting PDEBench DaRUS download...")
        result = _try_pdebench(N_SAMPLES)
        if result is not None:
            inputs, outputs = result
            provenance = (
                "PDEBench 2D Darcy Flow (beta=1.0), subset of first 100 samples, "
                "sourced from DaRUS doi:10.18419/darus-2986. MIT license."
            )
            print("  success.")
        else:
            print("  fell through to local solver.")

    if inputs is None or outputs is None:
        print(f"Solving 2D Darcy flow locally on a {GRID}x{GRID} grid for {N_SAMPLES} samples...")
        inputs, outputs = _solve_darcy_local(N_SAMPLES, GRID)
        provenance = (
            f"Local 5-point finite-difference solver for -div(a*grad u) = 1 on "
            f"[0,1]^2 with Dirichlet-0 boundary. Permeability a(x) drawn from a "
            f"log-Gaussian random field (length_scale=0.08, sigma=0.75, seed={SEED}). "
            f"Grid: {GRID}x{GRID}. This is real Darcy physics, solved locally; it "
            f"is NOT a copy of the PDEBench dataset."
        )

    np.savez_compressed(OUTPUT_NPZ, inputs=inputs, outputs=outputs)
    size_bytes = OUTPUT_NPZ.stat().st_size
    print(f"Wrote {OUTPUT_NPZ}  ({size_bytes} bytes, {size_bytes/1e6:.2f} MB)")
    print(f"  inputs.shape={inputs.shape}  dtype={inputs.dtype}")
    print(f"  outputs.shape={outputs.shape}  dtype={outputs.dtype}")

    OUTPUT_README.write_text(
        "# Darcy flow physics fixture\n\n"
        f"File: `darcy_small.npz` ({size_bytes} bytes)\n\n"
        "## Keys\n"
        f"* `inputs`  : float32 `(N={inputs.shape[0]}, H={inputs.shape[1]}, W={inputs.shape[2]})` "
        "permeability field a(x).\n"
        f"* `outputs` : float32 `(N={outputs.shape[0]}, H={outputs.shape[1]}, W={outputs.shape[2]})` "
        "pressure field u(x).\n\n"
        "## Provenance\n"
        f"{provenance}\n\n"
        "## Regenerating\n"
        "`python test_docs/fetch/download_pdebench_darcy.py`\n\n"
        "Pass `--force-local` to skip the download attempt and go straight to the local FD solver.\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_README}")


if __name__ == "__main__":
    sys.exit(main())
