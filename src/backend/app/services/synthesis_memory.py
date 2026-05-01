"""Synthesis Memory Service — Phase 4 of the Atlas Discovery OS.

After each experiment cycle (synthesis planning → spectrum verification → assay),
this service:
  1. Writes a SynthesisAttempt node + ATTEMPTED_VIA edge to the SQLite knowledge graph
  2. Embeds a human-readable summary into Qdrant so future search_literature calls
     automatically surface prior experiments for structurally similar molecules
  3. Provides a Tanimoto-similarity query to find past attempts on related compounds

This creates the "compound moat" described in the ConversionPlan §9:
  each experiment deposits long-lived memory that no cloud competitor can replicate.

All DB operations run in asyncio.run_in_executor() to avoid blocking the event loop.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SynthesisMemoryService:
    """
    Persists experimental results to the knowledge graph (SQLite) and
    the semantic vector store (Qdrant).

    The design follows the same "run sync DB work in executor" pattern used
    throughout retrieveal.py and graph.py.
    """

    # Node label and edge type written to the SQLite graph
    NODE_LABEL = "SynthesisAttempt"
    EDGE_TYPE = "ATTEMPTED_VIA"
    # Qdrant collection suffix to isolate synthesis memories from paper chunks
    SYNTH_COLLECTION = None  # Uses the main collection — tagged via payload

    def __init__(self, llm_service=None, qdrant_client=None):
        """
        Args:
            llm_service:   LLMService instance (for embed()). If None, a new
                           singleton is obtained lazily on first use.
            qdrant_client: Qdrant QdrantClient. If None, the global singleton
                           is obtained lazily.
        """
        self._llm_service = llm_service
        self._qdrant_client = qdrant_client

    # ------------------------------------------------------------------
    # Private: lazy service resolution
    # ------------------------------------------------------------------

    def _get_llm(self):
        if self._llm_service is None:
            from app.services.llm import LLMService
            self._llm_service = LLMService.get_instance()
        return self._llm_service

    def _get_qdrant(self, project_id: str):
        # Per-project; never cache across projects.
        from app.core.qdrant_store import get_qdrant_client
        return get_qdrant_client(project_id)

    # ------------------------------------------------------------------
    # Public API — record_experiment
    # ------------------------------------------------------------------

    async def record_experiment(
        self,
        project_id: str,
        smiles: str,
        route: Optional[Dict[str, Any]] = None,
        match_score: Optional[float] = None,
        assay_result: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> str:
        """
        Write a SynthesisAttempt node (and ATTEMPTED_VIA edge) to the SQLite
        knowledge graph.

        Args:
            project_id:   Project scope.
            smiles:       SMILES of the target molecule.
            route:        Synthesis route dict returned by plan_synthesis plugin.
                          May be None if synthesis planning was not performed.
            match_score:  NMR verification match score (0.0–1.0). None if not yet
                          verified.
            assay_result: Biological assay results, e.g. {"IC50_nM": 45}.
            notes:        Free-text notes from the researcher.

        Returns:
            The UUID string of the newly created SynthesisAttempt node.
        """
        node_id = str(uuid.uuid4())
        props = {
            "name": f"SynthesisAttempt:{smiles[:40]}",
            "smiles": smiles,
            "route": route or {},
            "match_score": match_score,
            "assay_result": assay_result or {},
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
        }

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._write_node_sync, node_id, project_id, props
        )

        logger.info(
            f"SynthesisAttempt recorded: node_id={node_id}, smiles={smiles[:40]}, "
            f"match_score={match_score}"
        )
        return node_id

    def _write_node_sync(
        self, node_id: str, project_id: str, props: Dict[str, Any]
    ) -> None:
        """Synchronous DB write — called inside run_in_executor."""
        from app.core.database import get_project_session, Node, Edge

        session = get_project_session(project_id)
        try:
            node = Node(
                id=node_id,
                label=self.NODE_LABEL,
                properties=props,
                project_id=project_id,
                document_id=None,  # synthesis memories are not tied to documents
            )
            session.add(node)

            # Find an existing compound node for the same SMILES (if any) and
            # create an ATTEMPTED_VIA edge linking it to this attempt.
            existing_compound = (
                session.query(Node)
                .filter(
                    Node.label == "chemical",
                    Node.project_id == project_id,
                )
                .all()
            )
            smiles_target = props["smiles"]
            for compound_node in existing_compound:
                node_smiles = (compound_node.properties or {}).get("smiles", "")
                if node_smiles == smiles_target:
                    edge = Edge(
                        id=str(uuid.uuid4()),
                        source_id=compound_node.id,
                        target_id=node_id,
                        type=self.EDGE_TYPE,
                        properties={"context": "Synthesis experiment recorded by Atlas Discovery OS"},
                        project_id=project_id,
                        document_id=None,
                    )
                    session.add(edge)
                    break

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to write SynthesisAttempt node: {e}")
            raise
        finally:
            session.close()

    # ------------------------------------------------------------------
    # Public API — embed_experiment
    # ------------------------------------------------------------------

    async def embed_experiment(
        self,
        project_id: str,
        experiment_node_id: str,
        smiles: str,
        route: Optional[Dict[str, Any]] = None,
        match_score: Optional[float] = None,
        assay_result: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> None:
        """
        Embed a human-readable experiment summary into Qdrant.

        The payload is tagged with ``source_type: "synthesis_memory"`` so that
        future search_literature calls using this project will naturally surface
        prior experiments alongside uploaded paper chunks.

        Args:
            project_id:          Project scope.
            experiment_node_id:  UUID of the SynthesisAttempt node (from record_experiment).
            smiles:              SMILES of the target molecule.
            route:               Synthesis route dict (optional).
            match_score:         NMR verification score (optional).
            assay_result:        Assay result dict (optional).
            notes:               Researcher notes (optional).
        """
        from app.core.qdrant_store import PROJECT_QDRANT_COLLECTION

        summary = self._build_summary(smiles, route, match_score, assay_result, notes)
        logger.info(f"Embedding experiment summary for node {experiment_node_id}")

        llm = self._get_llm()
        vector = await llm.embed(summary)

        qdrant = self._get_qdrant(project_id)
        collection = PROJECT_QDRANT_COLLECTION
        point_id = str(uuid.uuid4())

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._upsert_qdrant_sync,
            qdrant,
            collection,
            point_id,
            vector,
            {
                "text": summary,
                "doc_id": f"synth:{experiment_node_id}",
                "source_type": "synthesis_memory",
                "project_id": project_id,
                "node_id": experiment_node_id,
                "smiles": smiles,
                "match_score": match_score,
                "metadata": {
                    "filename": f"SynthesisAttempt:{smiles[:30]}",
                    "page": 0,
                    "source_type": "synthesis_memory",
                },
            },
        )
        logger.info(f"Experiment embedded into Qdrant: point_id={point_id}")

    def _upsert_qdrant_sync(
        self,
        qdrant_client,
        collection: str,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """Synchronous Qdrant upsert — called inside run_in_executor."""
        from qdrant_client.models import PointStruct

        qdrant_client.upsert(
            collection_name=collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

    # ------------------------------------------------------------------
    # Public API — find_similar_attempts
    # ------------------------------------------------------------------

    async def find_similar_attempts(
        self,
        project_id: str,
        smiles: str,
        top_k: int = 5,
        tanimoto_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Find past SynthesisAttempt nodes in this project whose target molecule
        has a Tanimoto fingerprint similarity ≥ tanimoto_threshold vs. `smiles`.

        Uses RDKit Morgan fingerprints (radius=2, 2048 bits) — the same standard
        used throughout cheminformatics for structural similarity.

        Args:
            project_id:          Project scope.
            smiles:              Query SMILES.
            top_k:               Maximum number of results to return.
            tanimoto_threshold:  Minimum Tanimoto coefficient (default 0.7).

        Returns:
            List of dicts, each with keys: node_id, smiles, match_score,
            assay_result, route, tanimoto, timestamp.
            Sorted descending by Tanimoto similarity.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._find_similar_sync,
            project_id,
            smiles,
            top_k,
            tanimoto_threshold,
        )

    def _find_similar_sync(
        self,
        project_id: str,
        query_smiles: str,
        top_k: int,
        threshold: float,
    ) -> List[Dict[str, Any]]:
        """Synchronous similarity search — called inside run_in_executor."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, DataStructs
        except ImportError:
            logger.warning("RDKit not available — find_similar_attempts returns empty")
            return []

        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            logger.warning(f"Invalid query SMILES: {query_smiles}")
            return []

        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, radius=2, nBits=2048)

        from app.core.database import get_project_session, Node

        session = get_project_session(project_id)
        try:
            existing = (
                session.query(Node)
                .filter(Node.label == self.NODE_LABEL)
                .all()
            )
        finally:
            session.close()

        results = []
        for node in existing:
            props = node.properties or {}
            candidate_smiles = props.get("smiles", "")
            if not candidate_smiles:
                continue

            cand_mol = Chem.MolFromSmiles(candidate_smiles)
            if cand_mol is None:
                continue

            cand_fp = AllChem.GetMorganFingerprintAsBitVect(
                cand_mol, radius=2, nBits=2048
            )
            tanimoto = DataStructs.TanimotoSimilarity(query_fp, cand_fp)

            if tanimoto >= threshold:
                results.append({
                    "node_id": str(node.id),
                    "smiles": candidate_smiles,
                    "match_score": props.get("match_score"),
                    "assay_result": props.get("assay_result", {}),
                    "route": props.get("route", {}),
                    "tanimoto": round(tanimoto, 4),
                    "timestamp": props.get("timestamp", ""),
                    "notes": props.get("notes", ""),
                })

        results.sort(key=lambda x: x["tanimoto"], reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        smiles: str,
        route: Optional[Dict[str, Any]],
        match_score: Optional[float],
        assay_result: Optional[Dict[str, Any]],
        notes: str,
    ) -> str:
        """Build a plaintext experiment summary for embedding."""
        parts = [
            f"Synthesis attempt for molecule: {smiles}.",
        ]

        if route:
            steps = route.get("steps", [])
            num_steps = len(steps) if isinstance(steps, list) else 0
            predicted_yield = route.get("predicted_yield", "unknown")
            if num_steps:
                parts.append(
                    f"Synthesis route: {num_steps} steps, predicted yield {predicted_yield}."
                )
            if route.get("reagents"):
                reagent_str = ", ".join(str(r) for r in route["reagents"][:5])
                parts.append(f"Key reagents: {reagent_str}.")
            if route.get("error"):
                parts.append(f"Route error: {route['error']}.")

        if match_score is not None:
            outcome = "PASSED" if match_score >= 0.7 else "FAILED"
            parts.append(
                f"NMR spectrum verification: match score {match_score:.2f} "
                f"({outcome}, threshold 0.70)."
            )

        if assay_result:
            for k, v in assay_result.items():
                parts.append(f"Assay result — {k}: {v}.")

        if notes:
            parts.append(f"Researcher notes: {notes}")

        return " ".join(parts)


# ============================================================
# Singleton accessor
# ============================================================

_synthesis_memory: Optional[SynthesisMemoryService] = None


def get_synthesis_memory() -> SynthesisMemoryService:
    """Get or create the singleton SynthesisMemoryService."""
    global _synthesis_memory
    if _synthesis_memory is None:
        _synthesis_memory = SynthesisMemoryService()
    return _synthesis_memory
