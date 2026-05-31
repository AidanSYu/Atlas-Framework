"""ADMET prediction plugin — fully self-contained, no app imports.

Predicts ADMET properties using admet_ai when available, or realistic mock
predictions based on RDKit descriptors or string-length heuristics.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _mock_admet_from_mw(mw: float) -> dict:
    if mw < 200:
        herg, dili, caco2, cyp3a4 = "LOW", "LOW", "HIGH", "LOW"
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
        herg, dili, caco2, cyp3a4 = "HIGH", "HIGH", "LOW", "HIGH"

    risk_factors = [herg, dili, cyp3a4]
    high_count = sum(1 for r in risk_factors if r == "HIGH")
    med_count = sum(1 for r in risk_factors if r == "MEDIUM")
    overall = "HIGH" if high_count >= 1 else ("MEDIUM" if med_count >= 1 else "LOW")
    return {"herg_risk": herg, "dili_risk": dili, "caco2_permeability": caco2,
            "cyp3a4_inhibition": cyp3a4, "overall_risk": overall}


def _mock_admet_from_length(smiles_len: int) -> dict:
    if smiles_len < 15:
        herg, dili, caco2, cyp3a4 = "LOW", "LOW", "HIGH", "LOW"
    elif smiles_len < 35:
        herg, dili, caco2, cyp3a4 = "LOW", "LOW", "MEDIUM", "MEDIUM"
    elif smiles_len < 55:
        herg, dili, caco2, cyp3a4 = "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM"
    else:
        herg, dili, caco2, cyp3a4 = "MEDIUM", "HIGH", "LOW", "HIGH"

    risk_factors = [herg, dili, cyp3a4]
    high_count = sum(1 for r in risk_factors if r == "HIGH")
    med_count = sum(1 for r in risk_factors if r == "MEDIUM")
    overall = "HIGH" if high_count >= 1 else ("MEDIUM" if med_count >= 1 else "LOW")
    return {"herg_risk": herg, "dili_risk": dili, "caco2_permeability": caco2,
            "cyp3a4_inhibition": cyp3a4, "overall_risk": overall}


class PredictAdmetWrapper:

    def __init__(self):
        self._model: Any = None
        self._attempted_load = False

    def _ensure_model(self) -> Any:
        if self._attempted_load:
            return self._model
        self._attempted_load = True
        try:
            from admet_ai import ADMETModel
            logger.info("Loading ADMET-AI model...")
            self._model = ADMETModel()
            logger.info("ADMET-AI model loaded.")
        except ImportError:
            logger.info("admet_ai not installed; will use mock ADMET predictions.")
        except Exception as e:
            logger.warning("Failed to load ADMET-AI model: %s. Using mock.", e)
        return self._model

    async def invoke(
        self,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        args = arguments or {}
        smiles_list = args.get("smiles_list", [])
        if not isinstance(smiles_list, list):
            smiles_list = [smiles_list] if smiles_list else []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._predict, smiles_list)

    def _predict(self, smiles_list: List[str]) -> dict:
        model = self._ensure_model()
        predictions: List[dict] = []

        if model is not None:
            try:
                raw = model.predict(smiles_list)
                for i, smi in enumerate(smiles_list):
                    pred = self._parse_admet_result(smi, raw, i)
                    pred["engine_used"] = "admet_ai"
                    pred["valid"] = True
                    predictions.append(pred)
            except Exception as e:
                logger.exception("admet_ai prediction failed, falling back to heuristic: %s", e)
                model = None
                predictions = []

        if model is None:
            for smi in smiles_list:
                predictions.append(self._mock_predict((smi or "").strip()))

        high_risk = sum(1 for p in predictions if p.get("overall_risk") == "HIGH")
        medium_risk = sum(1 for p in predictions if p.get("overall_risk") == "MEDIUM")
        low_risk = sum(1 for p in predictions if p.get("overall_risk") == "LOW")
        heuristic_count = sum(1 for p in predictions if p.get("engine_used", "").startswith("heuristic"))
        engine_note = (
            " WARNING: all results are SMILES-based heuristics, not real ADMET predictions — install admet_ai for trustworthy output."
            if heuristic_count == len(predictions) and predictions
            else (f" Note: {heuristic_count}/{len(predictions)} results are heuristic, not from admet_ai." if heuristic_count else "")
        )
        return {
            "predictions": predictions,
            "engine_used": "admet_ai" if heuristic_count == 0 else ("mixed" if heuristic_count < len(predictions) else "heuristic"),
            "valid": heuristic_count == 0,
            "summary": (
                f"ADMET predictions for {len(predictions)} molecules: "
                f"{low_risk} low risk, {medium_risk} medium, {high_risk} high." + engine_note
            ),
        }

    @staticmethod
    def _parse_admet_result(smi: str, raw: Any, index: int) -> dict:
        try:
            row = raw.iloc[index] if hasattr(raw, "iloc") else raw[index]
        except (IndexError, KeyError):
            return PredictAdmetWrapper._mock_predict(smi)

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

        risk_factors = [herg, dili, cyp3a4]
        high_count = sum(1 for r in risk_factors if r == "HIGH")
        med_count = sum(1 for r in risk_factors if r == "MEDIUM")
        overall = "HIGH" if high_count >= 1 else ("MEDIUM" if med_count >= 1 else "LOW")
        return {"smiles": smi, "herg_risk": herg, "dili_risk": dili,
                "caco2_permeability": caco2, "cyp3a4_inhibition": cyp3a4, "overall_risk": overall}

    @staticmethod
    def _mock_predict(smi: str) -> dict:
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
            engine = "heuristic_rdkit_mw"
        else:
            base = _mock_admet_from_length(len(smi))
            engine = "heuristic_smiles_length"
        return {"smiles": smi, **base, "engine_used": engine, "valid": False}


PLUGIN = PredictAdmetWrapper()
