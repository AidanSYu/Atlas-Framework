"""ADMET Prediction Plugin.

Predicts ADMET properties using admet_ai when available, or realistic mock
predictions based on RDKit descriptors or string-length heuristics.
"""
import asyncio
import logging
from typing import Any, List

from app.services.plugins.base import BasePlugin

logger = logging.getLogger(__name__)


def _mock_admet_from_mw(mw: float) -> dict:
    """Generate realistic mock ADMET from molecular weight heuristics.

    Note: caco2_permeability = HIGH means GOOD oral absorption (not a risk).
    Overall risk is determined by hERG, DILI, and CYP3A4 only.
    """
    if mw < 200:
        herg = "LOW"
        dili = "LOW"
        caco2 = "HIGH"   # good permeability for small molecules
        cyp3a4 = "LOW"
    elif mw < 350:
        herg = "LOW" if mw < 280 else "MEDIUM"
        dili = "LOW" if mw < 300 else "MEDIUM"
        caco2 = "HIGH" if mw < 300 else "MEDIUM"
        cyp3a4 = "LOW" if mw < 260 else "MEDIUM"
    elif mw < 500:
        herg = "MEDIUM" if mw < 450 else "HIGH"
        dili = "MEDIUM"
        caco2 = "MEDIUM" if mw < 450 else "LOW"
        cyp3a4 = "MEDIUM" if mw < 420 else "HIGH"
    else:
        herg = "HIGH"
        dili = "HIGH"
        caco2 = "LOW"
        cyp3a4 = "HIGH"

    # caco2 permeability: HIGH = good absorption, so exclude from risk tally
    risk_factors = [herg, dili, cyp3a4]
    high_count = sum(1 for r in risk_factors if r == "HIGH")
    med_count = sum(1 for r in risk_factors if r == "MEDIUM")
    overall = "HIGH" if high_count >= 1 else ("MEDIUM" if med_count >= 1 else "LOW")

    return {
        "herg_risk": herg,
        "dili_risk": dili,
        "caco2_permeability": caco2,
        "cyp3a4_inhibition": cyp3a4,
        "overall_risk": overall,
    }


def _mock_admet_from_length(smiles_len: int) -> dict:
    """Fallback mock ADMET from SMILES string length (no RDKit)."""
    if smiles_len < 15:
        herg, dili, caco2, cyp3a4 = "LOW", "LOW", "HIGH", "LOW"
    elif smiles_len < 35:
        herg, dili, caco2, cyp3a4 = "LOW", "LOW", "MEDIUM", "MEDIUM"
    elif smiles_len < 55:
        herg, dili, caco2, cyp3a4 = "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM"
    else:
        herg, dili, caco2, cyp3a4 = "MEDIUM", "HIGH", "LOW", "HIGH"

    # caco2 permeability: HIGH = good absorption, exclude from risk
    risk_factors = [herg, dili, cyp3a4]
    high_count = sum(1 for r in risk_factors if r == "HIGH")
    med_count = sum(1 for r in risk_factors if r == "MEDIUM")
    overall = "HIGH" if high_count >= 1 else ("MEDIUM" if med_count >= 1 else "LOW")

    return {
        "herg_risk": herg,
        "dili_risk": dili,
        "caco2_permeability": caco2,
        "cyp3a4_inhibition": cyp3a4,
        "overall_risk": overall,
    }


class ADMETPredictPlugin(BasePlugin):
    """Predict ADMET properties for SMILES using admet_ai or mock heuristics."""

    @property
    def name(self) -> str:
        return "predict_admet"

    @property
    def description(self) -> str:
        return (
            "Predict ADMET properties (hERG, DILI, Caco2 permeability, CYP3A4 inhibition) "
            "for a list of SMILES strings. Uses admet_ai when available, else mock heuristics."
        )

    async def load(self) -> Any:
        """Lazy-load admet_ai model if available."""
        try:
            from admet_ai import ADMETModel
            logger.info("Loading ADMET-AI model (may take a moment)...")
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(None, lambda: ADMETModel())
            logger.info("ADMET-AI model loaded.")
            return model
        except ImportError:
            logger.info("admet_ai not installed; will use mock ADMET predictions.")
            return None
        except Exception as e:
            logger.warning("Failed to load ADMET-AI model: %s. Using mock.", e)
            return None

    async def execute(self, model: Any, **kwargs) -> dict:
        """Predict ADMET properties for a list of SMILES.

        Args (in kwargs):
            smiles_list: list[str] -- SMILES strings to predict.

        Returns:
            Dict with predictions, summary.
        """
        smiles_list = kwargs.get("smiles_list", [])
        if not isinstance(smiles_list, list):
            smiles_list = [smiles_list] if smiles_list else []

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._predict, model, smiles_list)

    def _predict(self, model: Any, smiles_list: List[str]) -> dict:
        """Synchronous prediction logic."""
        predictions: List[dict] = []

        if model is not None:
            try:
                from admet_ai import ADMETModel
                admet = ADMETModel()
                raw = admet.predict(smiles_list)

                for i, smi in enumerate(smiles_list):
                    pred = self._parse_admet_result(smi, raw, i)
                    predictions.append(pred)
            except Exception as e:
                logger.exception("admet_ai prediction failed, falling back to mock: %s", e)
                model = None

        if model is None:
            for smi in smiles_list:
                smi = (smi or "").strip()
                pred = self._mock_predict(smi)
                predictions.append(pred)

        high_risk = sum(1 for p in predictions if p.get("overall_risk") == "HIGH")
        medium_risk = sum(1 for p in predictions if p.get("overall_risk") == "MEDIUM")
        low_risk = sum(1 for p in predictions if p.get("overall_risk") == "LOW")

        return {
            "predictions": predictions,
            "summary": (
                f"ADMET predictions for {len(predictions)} molecules: "
                f"{low_risk} low risk, {medium_risk} medium, {high_risk} high."
            ),
        }

    def _parse_admet_result(self, smi: str, raw: Any, index: int) -> dict:
        """Parse admet_ai output into our schema."""
        try:
            row = raw.iloc[index] if hasattr(raw, "iloc") else raw[index]
        except (IndexError, KeyError):
            return self._mock_predict(smi)

        def get_val(key: str, default: str = "LOW") -> str:
            val = row.get(key, default) if hasattr(row, "get") else getattr(row, key, default)
            if isinstance(val, (int, float)):
                if key in ("herg_ic50", "herg"):
                    return "HIGH" if val < 1 else ("MEDIUM" if val < 10 else "LOW")
                if key in ("dili", "dili_prob"):
                    return "HIGH" if val > 0.7 else ("MEDIUM" if val > 0.4 else "LOW")
                if "caco2" in key.lower():
                    return "HIGH" if val > 1e-5 else ("MEDIUM" if val > 1e-6 else "LOW")
                if "cyp3a4" in key.lower():
                    return "HIGH" if val > 0.7 else ("MEDIUM" if val > 0.4 else "LOW")
            if isinstance(val, str) and val.upper() in ("LOW", "MEDIUM", "HIGH"):
                return val.upper()
            return str(default)

        herg = get_val("herg_risk", get_val("herg", get_val("herg_ic50", "LOW")))
        dili = get_val("dili_risk", get_val("dili", "LOW"))
        caco2 = get_val("caco2_permeability", get_val("caco2", "MEDIUM"))
        cyp3a4 = get_val("cyp3a4_inhibition", get_val("cyp3a4", "LOW"))

        # caco2 permeability: HIGH = good absorption, exclude from risk tally
        risk_factors = [herg, dili, cyp3a4]
        high_count = sum(1 for r in risk_factors if r == "HIGH")
        med_count = sum(1 for r in risk_factors if r == "MEDIUM")
        overall = "HIGH" if high_count >= 1 else ("MEDIUM" if med_count >= 1 else "LOW")

        return {
            "smiles": smi,
            "herg_risk": herg,
            "dili_risk": dili,
            "caco2_permeability": caco2,
            "cyp3a4_inhibition": cyp3a4,
            "overall_risk": overall,
        }

    def _mock_predict(self, smi: str) -> dict:
        """Mock prediction using RDKit MW or string length."""
        smi = (smi or "").strip()
        mw = None

        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
        except Exception:
            pass

        if mw is not None:
            base = _mock_admet_from_mw(mw)
        else:
            base = _mock_admet_from_length(len(smi))

        return {
            "smiles": smi,
            "herg_risk": base["herg_risk"],
            "dili_risk": base["dili_risk"],
            "caco2_permeability": base["caco2_permeability"],
            "cyp3a4_inhibition": base["cyp3a4_inhibition"],
            "overall_risk": base["overall_risk"],
        }

    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "smiles_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of SMILES strings for ADMET prediction",
                },
            },
            "required": ["smiles_list"],
        }

    def output_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "smiles": {"type": "string"},
                            "herg_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                            "dili_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                            "caco2_permeability": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                            "cyp3a4_inhibition": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                            "overall_risk": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                        },
                    },
                },
                "summary": {"type": "string"},
            },
            "required": ["predictions", "summary"],
        }
