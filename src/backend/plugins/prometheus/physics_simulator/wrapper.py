"""Physics surrogate model plugin — fully self-contained, no app imports.

Supports Fourier Neural Operator (FNO) via neuraloperator as default and
Physics-Informed Neural Network (PINN) as a pure-PyTorch path.  Includes
MC-dropout uncertainty quantification, Mahalanobis OOD rejection, and a
built-in solder reflow heat transfer demo for self-testing.
"""
import asyncio
import logging
import math
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard dependency gate — refuse clearly if torch is missing
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import numpy as np
except ImportError as _exc:
    raise ImportError(
        "physics_simulator plugin requires PyTorch and NumPy. "
        "Install them with:  pip install torch numpy"
    ) from _exc

# Optional FNO dependency — detected at call time
_HAS_NEURALOP = False
try:
    from neuralop.models import FNO  # type: ignore[import-untyped]
    _HAS_NEURALOP = True
except ImportError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# PINN architecture for 1D/2D heat equation  u_t = alpha * u_xx
# ---------------------------------------------------------------------------
class PINN(nn.Module):
    """Physics-Informed Neural Network with MC-dropout layers."""

    def __init__(self, hidden: int = 64, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        net: List[nn.Module] = [nn.Linear(2, hidden), nn.Tanh(), nn.Dropout(dropout)]
        for _ in range(layers - 1):
            net.extend([nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(dropout)])
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=-1))


# ---------------------------------------------------------------------------
# OOD detector via Mahalanobis distance
# ---------------------------------------------------------------------------
class MahalanobisOOD:
    """Tracks training-set feature statistics and scores new inputs."""

    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None
        self.cov_inv: Optional[torch.Tensor] = None
        self.threshold: float = 0.0

    def fit(self, features: torch.Tensor) -> None:
        """Compute mean and inverse covariance from flattened training features."""
        flat = features.reshape(features.shape[0], -1).float()
        self.mean = flat.mean(dim=0)
        centered = flat - self.mean
        cov = (centered.T @ centered) / max(flat.shape[0] - 1, 1)
        # Regularise for numerical stability
        cov += 1e-6 * torch.eye(cov.shape[0], device=cov.device)
        self.cov_inv = torch.linalg.inv(cov)
        # Chi-squared 99th percentile approximation for p dimensions
        p = float(flat.shape[1])
        self.threshold = p + 2.326 * math.sqrt(2.0 * p)

    def score(self, features: torch.Tensor) -> float:
        """Return mean Mahalanobis distance for input (single or batch)."""
        if self.mean is None or self.cov_inv is None:
            return 0.0
        flat = features.float()
        if flat.ndim == 1:
            flat = flat.unsqueeze(0)
        if flat.ndim > 2:
            flat = flat.reshape(flat.shape[0], -1)
        # Ensure feature dimension matches OOD detector's training dimension
        expected_dim = self.mean.shape[0]
        if flat.shape[1] != expected_dim:
            # Truncate or pad to match — defensive against shape mismatches
            if flat.shape[1] > expected_dim:
                flat = flat[:, :expected_dim]
            else:
                pad = torch.zeros(flat.shape[0], expected_dim - flat.shape[1])
                flat = torch.cat([flat, pad], dim=1)
        diff = flat - self.mean.unsqueeze(0)
        per_sample = torch.sqrt(((diff @ self.cov_inv) * diff).sum(dim=1))
        return float(per_sample.mean())

    def is_ood(self, features: torch.Tensor) -> tuple:
        """Return (distance, rejected_flag)."""
        d = self.score(features)
        return d, d > self.threshold


# ---------------------------------------------------------------------------
# MC-dropout inference helper
# ---------------------------------------------------------------------------
def _mc_predict(model: nn.Module, forward_fn, n_samples: int) -> tuple:
    """Run *n_samples* stochastic forward passes with dropout enabled.

    Returns (mean_prediction, std_prediction) as numpy arrays.
    """
    model.train()  # keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            preds.append(forward_fn().cpu())
    stacked = torch.stack(preds)
    mean = stacked.mean(dim=0).numpy()
    std = stacked.std(dim=0).numpy()
    return mean, std


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------
class PhysicsSimulatorWrapper:

    def __init__(self) -> None:
        self._pinn: Optional[nn.Module] = None
        self._fno: Optional[nn.Module] = None
        self._ood: MahalanobisOOD = MahalanobisOOD()
        self._model_type: Optional[str] = None

    # ----- public entry point ---------------------------------------------
    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        mode = args.get("mode", "self_test")
        loop = asyncio.get_running_loop()

        if mode == "self_test":
            return await loop.run_in_executor(None, self._self_test, args)
        elif mode == "train":
            return await loop.run_in_executor(None, self._train, args)
        elif mode == "predict":
            return await loop.run_in_executor(None, self._predict, args)
        elif mode == "reflow_defect_physics":
            return await loop.run_in_executor(None, self._reflow_defect_physics, args)
        else:
            return {"error": f"Unknown mode '{mode}'. Use 'predict', 'train', 'self_test', or 'reflow_defect_physics'."}

    # ----- resolve model type ---------------------------------------------
    @staticmethod
    def _resolve_model_type(requested: str) -> str:
        if requested == "fno":
            if not _HAS_NEURALOP:
                raise RuntimeError(
                    "neuraloperator is not installed. "
                    "Install with:  pip install neuraloperator  — or use model_type='pinn'."
                )
            return "fno"
        if requested == "pinn":
            return "pinn"
        # auto: prefer FNO, fall back to PINN
        return "fno" if _HAS_NEURALOP else "pinn"

    # ----- training-data loader ------------------------------------------
    @staticmethod
    def _load_training_data_from_path(path: str) -> Dict[str, Any]:
        """Load an .npz with required keys 'inputs' and 'outputs'.

        Fails LOUDLY (raises) on a missing file or missing keys so the
        orchestrator sees a clear error rather than silently running on
        synthetic data. Additive — callers that provide training_data in-memory
        bypass this entirely.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"training_data_path does not exist or is not a file: {path}"
            )
        try:
            archive = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise ValueError(
                f"Failed to read training_data_path '{path}' as an .npz: {exc}"
            ) from exc
        missing = [k for k in ("inputs", "outputs") if k not in archive.files]
        if missing:
            raise KeyError(
                f"training_data_path '{path}' is missing required key(s): "
                f"{missing}. Found keys: {list(archive.files)}."
            )
        inputs = archive["inputs"]
        outputs = archive["outputs"]
        return {"inputs": inputs, "outputs": outputs}

    # ----- train ----------------------------------------------------------
    def _train(self, args: dict) -> dict:
        requested = args.get("model_type", "auto")
        try:
            mtype = self._resolve_model_type(requested)
        except RuntimeError as e:
            return {"error": str(e)}

        training_data = args.get("training_data")
        training_data_path = args.get("training_data_path")
        if training_data_path:
            if training_data:
                return {"error": (
                    "Pass either training_data OR training_data_path, not both. "
                    "Refusing to silently pick one."
                )}
            try:
                training_data = self._load_training_data_from_path(training_data_path)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                # Loud failure — do NOT fall back to synthetic data.
                return {"error": f"training_data_path load failed: {exc}"}
        if not training_data or "inputs" not in training_data or "outputs" not in training_data:
            return {"error": "training_data with 'inputs' and 'outputs' arrays (or a valid training_data_path) is required for train mode."}

        inputs = torch.tensor(np.array(training_data["inputs"]), dtype=torch.float32).to(DEVICE)
        outputs = torch.tensor(np.array(training_data["outputs"]), dtype=torch.float32).to(DEVICE)
        params = args.get("parameters", {})
        n_epochs = int(params.get("epochs", 500))
        lr = float(params.get("lr", 1e-3))

        if mtype == "fno":
            return self._train_fno(inputs, outputs, n_epochs, lr)
        else:
            return self._train_pinn(inputs, outputs, n_epochs, lr, params)

    def _train_fno(self, inputs: torch.Tensor, outputs: torch.Tensor,
                   n_epochs: int, lr: float) -> dict:
        # inputs: (B, C, H, W), outputs: (B, C, H, W)
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(1)
        if outputs.ndim == 3:
            outputs = outputs.unsqueeze(1)

        model = FNO(
            n_modes=(16, 16),
            in_channels=inputs.shape[1],
            out_channels=outputs.shape[1],
            hidden_channels=32,
            n_layers=4,
        ).to(DEVICE)

        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        losses: List[float] = []

        for epoch in range(n_epochs):
            optimiser.zero_grad()
            pred = model(inputs)
            loss = loss_fn(pred, outputs)
            loss.backward()
            optimiser.step()
            if epoch % max(1, n_epochs // 10) == 0:
                losses.append(round(loss.item(), 6))

        self._fno = model
        self._model_type = "fno"
        # Fit OOD detector on training inputs
        self._ood.fit(inputs.detach().cpu())

        return {
            "valid": True,
            "prediction": None,
            "uncertainty": None,
            "ood_score": None,
            "ood_rejected": None,
            "model_info": {
                "type": "fno",
                "device": str(DEVICE),
                "parameters": sum(p.numel() for p in model.parameters()),
                "epochs": n_epochs,
                "final_loss": losses[-1] if losses else None,
                "loss_history": losses,
            },
            "summary": (
                f"FNO trained for {n_epochs} epochs on {inputs.shape[0]} samples. "
                f"Final loss: {losses[-1] if losses else 'N/A'}."
            ),
        }

    def _train_pinn(self, inputs: torch.Tensor, outputs: torch.Tensor,
                    n_epochs: int, lr: float, params: dict) -> dict:
        alpha = float(params.get("alpha", 0.01))
        hidden = int(params.get("hidden", 64))
        layers = int(params.get("layers", 4))

        model = PINN(hidden=hidden, layers=layers).to(DEVICE)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)
        losses: List[float] = []

        # Expect inputs shape (N, 2) where col0=x, col1=t
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(-1)
        if inputs.shape[-1] < 2:
            return {"error": "PINN training inputs must have at least 2 columns (x, t)."}

        x_col = inputs[:, 0:1].requires_grad_(True)
        t_col = inputs[:, 1:2].requires_grad_(True)
        target = outputs.reshape(-1, 1)

        for epoch in range(n_epochs):
            optimiser.zero_grad()
            u = model(x_col, t_col)

            # Physics residual: u_t - alpha * u_xx
            u_t = torch.autograd.grad(u, t_col, grad_outputs=torch.ones_like(u),
                                      create_graph=True)[0]
            u_x = torch.autograd.grad(u, x_col, grad_outputs=torch.ones_like(u),
                                      create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x_col, grad_outputs=torch.ones_like(u_x),
                                       create_graph=True)[0]

            physics_loss = nn.functional.mse_loss(u_t, alpha * u_xx)
            data_loss = nn.functional.mse_loss(u, target)
            loss = data_loss + physics_loss

            loss.backward()
            optimiser.step()
            if epoch % max(1, n_epochs // 10) == 0:
                losses.append(round(loss.item(), 6))

        self._pinn = model
        self._model_type = "pinn"
        self._ood.fit(inputs.detach().cpu())

        return {
            "valid": True,
            "prediction": None,
            "uncertainty": None,
            "ood_score": None,
            "ood_rejected": None,
            "model_info": {
                "type": "pinn",
                "device": str(DEVICE),
                "parameters": sum(p.numel() for p in model.parameters()),
                "epochs": n_epochs,
                "final_loss": losses[-1] if losses else None,
                "loss_history": losses,
            },
            "summary": (
                f"PINN trained for {n_epochs} epochs on {inputs.shape[0]} points. "
                f"Final loss: {losses[-1] if losses else 'N/A'}."
            ),
        }

    # ----- predict --------------------------------------------------------
    def _predict(self, args: dict) -> dict:
        n_mc = int(args.get("n_mc_samples", 20))

        if self._model_type == "fno" and self._fno is not None:
            return self._predict_fno(args, n_mc)
        elif self._model_type == "pinn" and self._pinn is not None:
            return self._predict_pinn(args, n_mc)
        else:
            return {"error": "No trained model available. Run 'train' or 'self_test' first."}

    def _predict_fno(self, args: dict, n_mc: int) -> dict:
        inp = args.get("boundary_conditions", {}).get("input_field")
        if inp is None:
            return {"error": "boundary_conditions.input_field is required for FNO prediction."}
        tensor_in = torch.tensor(np.array(inp), dtype=torch.float32).to(DEVICE)
        if tensor_in.ndim == 2:
            tensor_in = tensor_in.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif tensor_in.ndim == 3:
            tensor_in = tensor_in.unsqueeze(0)

        model = self._fno
        mean, std = _mc_predict(model, lambda: model(tensor_in).squeeze(), n_mc)
        ood_dist, ood_flag = self._ood.is_ood(tensor_in.detach().cpu())

        return {
            "valid": True,
            "prediction": mean.tolist(),
            "uncertainty": std.tolist(),
            "ood_score": round(ood_dist, 4),
            "ood_rejected": ood_flag,
            "model_info": {"type": "fno", "device": str(DEVICE)},
            "summary": (
                f"FNO prediction with {n_mc} MC-dropout samples. "
                f"Mean uncertainty: {float(std.mean()):.4f}. "
                f"OOD score: {ood_dist:.2f} ({'REJECTED' if ood_flag else 'accepted'})."
            ),
        }

    def _predict_pinn(self, args: dict, n_mc: int) -> dict:
        bc = args.get("boundary_conditions", {})
        x_vals = bc.get("x")
        t_vals = bc.get("t")
        if x_vals is None or t_vals is None:
            return {"error": "boundary_conditions.x and boundary_conditions.t are required for PINN prediction."}

        x_t = torch.tensor(np.array(x_vals), dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        t_t = torch.tensor(np.array(t_vals), dtype=torch.float32).reshape(-1, 1).to(DEVICE)
        inp_flat = torch.cat([x_t, t_t], dim=-1)

        model = self._pinn
        mean, std = _mc_predict(model, lambda: model(x_t, t_t).squeeze(), n_mc)
        ood_dist, ood_flag = self._ood.is_ood(inp_flat.detach().cpu())

        return {
            "valid": True,
            "prediction": mean.tolist(),
            "uncertainty": std.tolist(),
            "ood_score": round(ood_dist, 4),
            "ood_rejected": ood_flag,
            "model_info": {"type": "pinn", "device": str(DEVICE)},
            "summary": (
                f"PINN prediction with {n_mc} MC-dropout samples. "
                f"Mean uncertainty: {float(std.mean()):.4f}. "
                f"OOD score: {ood_dist:.2f} ({'REJECTED' if ood_flag else 'accepted'})."
            ),
        }

    # ----- self-test: 2D solder reflow heat transfer ----------------------
    def _self_test(self, args: dict) -> dict:
        logger.info("physics_simulator self_test: generating synthetic solder reflow data")
        n_mc = int(args.get("n_mc_samples", 20))

        # --- Step 1: generate synthetic 2D heat equation data -------------
        Nx, Ny, Nt = 64, 64, 50
        u = np.zeros((Nt, Nx, Ny))
        u[0, 28:36, 28:36] = 250.0  # solder pad initial temperature
        alpha = 0.01

        for t in range(Nt - 1):
            lap = (
                np.roll(u[t], 1, 0) + np.roll(u[t], -1, 0)
                + np.roll(u[t], 1, 1) + np.roll(u[t], -1, 1)
                - 4 * u[t]
            )
            u[t + 1] = u[t] + alpha * lap

        # --- Step 2: prepare training pairs for PINN ---------------------
        # Sample (x, t) -> u(x, t) from the grid solution
        n_train = 2000
        rng = np.random.default_rng(42)
        xi = rng.integers(0, Nx, size=n_train)
        yi = rng.integers(0, Ny, size=n_train)
        ti = rng.integers(0, Nt, size=n_train)

        # Normalise spatial coords to [0, 1] and time to [0, 1]
        x_norm = xi.astype(np.float32) / (Nx - 1)
        t_norm = ti.astype(np.float32) / (Nt - 1)
        u_vals = np.array([u[ti[i], xi[i], yi[i]] for i in range(n_train)], dtype=np.float32)
        # Normalise target to [0, 1]
        u_max = max(float(u_vals.max()), 1e-8)
        u_vals_norm = u_vals / u_max

        inputs_np = np.stack([x_norm, t_norm], axis=-1)  # (N, 2)
        outputs_np = u_vals_norm  # (N,)

        inputs_t = torch.tensor(inputs_np, dtype=torch.float32).to(DEVICE)
        outputs_t = torch.tensor(outputs_np, dtype=torch.float32).to(DEVICE)

        # --- Step 3: train a small PINN for ~500 iterations ---------------
        model = PINN(hidden=64, layers=4).to(DEVICE)
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses: List[float] = []

        x_col = inputs_t[:, 0:1].requires_grad_(True)
        t_col = inputs_t[:, 1:2].requires_grad_(True)
        target = outputs_t.reshape(-1, 1)

        for epoch in range(500):
            optimiser.zero_grad()
            pred = model(x_col, t_col)

            # Physics residual
            u_t_grad = torch.autograd.grad(pred, t_col, grad_outputs=torch.ones_like(pred),
                                           create_graph=True)[0]
            u_x_grad = torch.autograd.grad(pred, x_col, grad_outputs=torch.ones_like(pred),
                                           create_graph=True)[0]
            u_xx_grad = torch.autograd.grad(u_x_grad, x_col, grad_outputs=torch.ones_like(u_x_grad),
                                            create_graph=True)[0]

            physics_loss = nn.functional.mse_loss(u_t_grad, alpha * u_xx_grad)
            data_loss = nn.functional.mse_loss(pred, target)
            loss = data_loss + physics_loss
            loss.backward()
            optimiser.step()

            if epoch % 50 == 0:
                losses.append(round(loss.item(), 6))

        self._pinn = model
        self._model_type = "pinn"

        # Fit OOD detector on training inputs
        self._ood.fit(inputs_t.detach().cpu())

        # --- Step 4: predict on test input with UQ -----------------------
        n_test = 100
        x_test = np.linspace(0, 1, n_test, dtype=np.float32)
        t_test = np.full(n_test, 0.5, dtype=np.float32)
        x_test_t = torch.tensor(x_test).reshape(-1, 1).to(DEVICE)
        t_test_t = torch.tensor(t_test).reshape(-1, 1).to(DEVICE)

        mean_pred, std_pred = _mc_predict(
            model, lambda: model(x_test_t, t_test_t).squeeze(), n_mc
        )

        test_input_flat = torch.cat([x_test_t.cpu(), t_test_t.cpu()], dim=-1)
        ood_dist_normal, ood_flag_normal = self._ood.is_ood(test_input_flat)

        # --- Step 5: check OOD on wildly different input ------------------
        x_wild = np.full(n_test, 50.0, dtype=np.float32)  # far outside [0,1]
        t_wild = np.full(n_test, -10.0, dtype=np.float32)
        wild_flat = torch.tensor(np.stack([x_wild, t_wild], axis=-1))
        ood_dist_wild, ood_flag_wild = self._ood.is_ood(wild_flat)

        return {
            "valid": True,
            "prediction": mean_pred.tolist(),
            "uncertainty": std_pred.tolist(),
            "ood_score": round(ood_dist_normal, 4),
            "ood_rejected": ood_flag_normal,
            "wild_ood_score": round(ood_dist_wild, 4),
            "wild_ood_rejected": ood_flag_wild,
            "mean_uncertainty": round(float(std_pred.mean()), 6),
            "model_info": {
                "type": "pinn",
                "device": str(DEVICE),
                "parameters": sum(p.numel() for p in model.parameters()),
                "epochs": 500,
                "final_loss": losses[-1] if losses else None,
                "loss_history": losses,
                "training_points": n_train,
                "grid_size": f"{Nx}x{Ny}x{Nt}",
                "diffusivity": alpha,
            },
            "summary": (
                f"Self-test complete: heat-diffusion surrogate sampled from a {Nx}x{Ny} grid over {Nt} timesteps. "
                f"PINN trained for 500 epochs on {n_train} sampled points (final loss: {losses[-1] if losses else 'N/A'}). "
                f"Test prediction mean uncertainty: {float(std_pred.mean()):.4f}. "
                f"In-distribution OOD score: {ood_dist_normal:.2f} ({'REJECTED' if ood_flag_normal else 'accepted'}). "
                f"Wild OOD score: {ood_dist_wild:.2f} ({'REJECTED' if ood_flag_wild else 'accepted'})."
            ),
        }


    # ----- reflow defect physics -----------------------------------------
    # SAC305 reference thresholds (Celsius + seconds). Source: IPC J-STD-020.
    # These drive the physics-consistent defect labels below.
    _LIQUIDUS_C = 217.0
    _PEAK_MIN_C = 235.0          # below this, wetting is incomplete -> cold joint
    _PEAK_MAX_C = 260.0          # above this, component damage / overshoot
    _TAL_MIN_S = 40.0             # time above liquidus minimum for good joints
    _TAL_MAX_S = 90.0             # excessive TAL -> intermetallic growth + voids
    _RAMP_MAX_C_PER_S = 3.0       # thermal shock threshold

    @staticmethod
    def _simulate_reflow_curve(
        peak_c: float,
        tal_s: float,
        ramp_c_per_s: float,
        duration_s: float = 300.0,
        dt_s: float = 1.0,
    ) -> "np.ndarray":
        """Generate a SAC305-shaped reflow curve from three physics knobs.

        The curve is piecewise-linear with four segments (preheat, soak,
        reflow-up + optional peak-hold, reflow-down), so `max |dT/dt|` exactly
        equals the requested `ramp_c_per_s`. TAL equals the requested `tal_s`
        as long as the peak is tall enough to contain it; otherwise TAL is
        capped at `2*(peak_c - liquidus)/ramp_c_per_s` (the physics ceiling).
        """
        n = max(2, int(duration_s / dt_s))
        t = np.arange(n, dtype=np.float32) * dt_s
        ambient_c = 25.0
        soak_c = 160.0
        soak_duration_s = 90.0
        ramp = max(ramp_c_per_s, 0.05)
        liq = PhysicsSimulatorWrapper._LIQUIDUS_C

        # Segment durations.
        preheat_s = max(0.0, (soak_c - ambient_c) / ramp)
        up_s = max(0.0, (peak_c - soak_c) / ramp) if peak_c > soak_c else 0.0
        down_s = max(0.0, (peak_c - ambient_c) / ramp) if peak_c > ambient_c else 0.0
        # Triangular TAL if no hold (TAL = 2 * (peak - liq)/ramp).
        triangle_tal_s = max(0.0, 2.0 * (peak_c - liq) / ramp) if peak_c > liq else 0.0
        hold_s = max(0.0, tal_s - triangle_tal_s)

        # Segment boundaries (cumulative time).
        t_preheat_end = preheat_s
        t_soak_end = t_preheat_end + soak_duration_s
        t_up_end = t_soak_end + up_s
        t_hold_end = t_up_end + hold_s
        t_down_end = t_hold_end + down_s

        curve = np.full(n, ambient_c, dtype=np.float32)
        for i, ti in enumerate(t):
            if ti <= t_preheat_end:
                curve[i] = ambient_c + ramp * ti
            elif ti <= t_soak_end:
                curve[i] = soak_c
            elif ti <= t_up_end:
                curve[i] = soak_c + ramp * (ti - t_soak_end)
            elif ti <= t_hold_end:
                curve[i] = peak_c
            elif ti <= t_down_end:
                curve[i] = peak_c - ramp * (ti - t_hold_end)
            else:
                curve[i] = ambient_c
        return curve

    @staticmethod
    def _classify_reflow_curve(curve: "np.ndarray", dt_s: float = 1.0) -> Dict[str, Any]:
        """Return the defect label + evidence derived purely from the curve.

        This is the physics ground truth that the renderer and the VLM alignment
        pipeline both consume. We grade against SAC305 thresholds.
        """
        peak_c = float(curve.max())
        above_liq = curve >= PhysicsSimulatorWrapper._LIQUIDUS_C
        tal_s = float(above_liq.sum() * dt_s)
        # Peak ramp rate across the whole curve (both directions).
        ramp = np.abs(np.diff(curve)) / dt_s
        max_ramp = float(ramp.max()) if len(ramp) else 0.0

        labels: List[str] = []
        if peak_c < PhysicsSimulatorWrapper._PEAK_MIN_C:
            labels.append("cold_joint")
        if peak_c > PhysicsSimulatorWrapper._PEAK_MAX_C:
            labels.append("overshoot_damage")
        if tal_s < PhysicsSimulatorWrapper._TAL_MIN_S and peak_c >= PhysicsSimulatorWrapper._LIQUIDUS_C:
            labels.append("insufficient_wetting")
        if tal_s > PhysicsSimulatorWrapper._TAL_MAX_S:
            labels.append("excessive_tal_voids")
        if max_ramp > PhysicsSimulatorWrapper._RAMP_MAX_C_PER_S:
            labels.append("thermal_shock_tombstone")

        primary = labels[0] if labels else "pass"
        return {
            "primary_defect": primary,
            "all_defect_labels": labels,
            "peak_c": round(peak_c, 2),
            "time_above_liquidus_s": round(tal_s, 2),
            "max_ramp_c_per_s": round(max_ramp, 3),
            "pass_conditions_met": not labels,
        }

    def _reflow_defect_physics(self, args: dict) -> dict:
        """Generate a batch of parametrised reflow profiles + physics labels.

        Callers feed `count` and optional `profile_spec` (a list of explicit
        peak/tal/ramp triples). Omitted entries are drawn from a deterministic
        grid that covers the good envelope plus each injected failure mode.
        """
        count = int(args.get("count", 12))
        seed = int(args.get("seed", 17))
        rng = np.random.default_rng(seed)
        dt_s = float(args.get("dt_s", 1.0))
        duration_s = float(args.get("duration_s", 300.0))

        explicit: List[Dict[str, float]] = list(args.get("profile_spec") or [])
        # Defect mix — even coverage across pass + each failure mode so the
        # downstream DPO dataset is balanced.
        mix_templates = [
            {"kind": "pass", "peak_c": 245.0, "tal_s": 60.0, "ramp_c_per_s": 1.5},
            {"kind": "cold_joint", "peak_c": 225.0, "tal_s": 30.0, "ramp_c_per_s": 1.5},
            {"kind": "overshoot_damage", "peak_c": 272.0, "tal_s": 65.0, "ramp_c_per_s": 1.8},
            {"kind": "insufficient_wetting", "peak_c": 230.0, "tal_s": 18.0, "ramp_c_per_s": 1.5},
            {"kind": "excessive_tal_voids", "peak_c": 250.0, "tal_s": 110.0, "ramp_c_per_s": 1.5},
            {"kind": "thermal_shock_tombstone", "peak_c": 245.0, "tal_s": 55.0, "ramp_c_per_s": 4.5},
        ]

        profiles: List[Dict[str, Any]] = []
        for idx in range(count):
            if idx < len(explicit):
                spec = explicit[idx]
            else:
                base = mix_templates[idx % len(mix_templates)]
                # Jitter around each template so consecutive calls produce variety.
                jitter = float(rng.normal(0.0, 1.0))
                spec = {
                    "peak_c": float(base["peak_c"]) + jitter * 2.0,
                    "tal_s": max(5.0, float(base["tal_s"]) + jitter * 3.0),
                    "ramp_c_per_s": max(0.2, float(base["ramp_c_per_s"]) + jitter * 0.15),
                }

            curve = self._simulate_reflow_curve(
                peak_c=float(spec["peak_c"]),
                tal_s=float(spec["tal_s"]),
                ramp_c_per_s=float(spec["ramp_c_per_s"]),
                duration_s=duration_s,
                dt_s=dt_s,
            )
            evidence = self._classify_reflow_curve(curve, dt_s=dt_s)
            profiles.append({
                "profile_id": f"rf_{idx:03d}",
                "requested": spec,
                "curve": curve.tolist(),
                "dt_s": dt_s,
                **evidence,
            })

        counts: Dict[str, int] = {}
        for p in profiles:
            counts[p["primary_defect"]] = counts.get(p["primary_defect"], 0) + 1

        return {
            "valid": True,
            "profiles": profiles,
            "label_counts": counts,
            "thresholds": {
                "liquidus_c": self._LIQUIDUS_C,
                "peak_min_c": self._PEAK_MIN_C,
                "peak_max_c": self._PEAK_MAX_C,
                "tal_min_s": self._TAL_MIN_S,
                "tal_max_s": self._TAL_MAX_S,
                "ramp_max_c_per_s": self._RAMP_MAX_C_PER_S,
            },
            "summary": (
                f"Generated {len(profiles)} reflow profiles with physics labels: "
                + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            ),
        }


PLUGIN = PhysicsSimulatorWrapper()
