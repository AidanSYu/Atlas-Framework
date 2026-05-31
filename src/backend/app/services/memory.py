"""Cross-campaign memory — the compounding substrate.

Writes one ``CampaignMemory`` row when a campaign completes, and exposes recall
over those rows. This is the "campaign N+1 beats campaign N" moat: a later
campaign reuses a prior plan skeleton and SKIPS the dead-ends an earlier one
already ruled out (the ARROWS3-style search-space pruning that makes the second
run reach a result in fewer steps).

v1 recall is keyword + structured (token-overlap × recency-decay × access) over
the local SQLite store — fully offline, no LLM on the hot path, and an honest,
strong baseline (simple keyword retrieval has been shown to match heavyweight
agent-memory frameworks). A semantic (Qdrant) signal and graph PPR are clean
later extensions; recall is structured so they slot in without changing the tool
contract. Every recalled row is an unverified hypothesis with provenance
(confidence flag + source campaign) so bad data is down-ranked, not compounded.
"""
from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.database import CampaignMemory, get_project_session

logger = logging.getLogger(__name__)

# Event-type string values (EventType is a str-Enum, so plain strings match).
_TOOL_INTENT = "TOOL_CALL_INTENT"
_TOOL_RESULT = "TOOL_EXECUTION_RESULT"
_FINAL_ANSWER = "FINAL_ANSWER"

_RECENCY_HALFLIFE_DAYS = 30.0  # Ebbinghaus-style decay; recent campaigns rank higher
_MAX_KEY_FACTS = 8
_MAX_DEAD_ENDS = 8
_CITATION_KEY_HINTS = ("citation", "citations", "source", "doi", "pmid", "chembl", "pdb")

_STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "your", "you",
    "are", "was", "were", "has", "have", "had", "will", "should", "can", "use",
    "using", "run", "find", "a", "an", "of", "to", "in", "on", "as", "is", "it",
    "evaluate", "molecule", "target", "candidate",
}


def _tokenize(text: str) -> set:
    return {
        t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(t) > 2 and t not in _STOPWORDS
    }


def _event_field(evt: Any, attr: str, default: Any = None) -> Any:
    """Read an attribute from a TaskEventDTO (object) or a plain dict event."""
    if isinstance(evt, dict):
        return evt.get(attr, default)
    return getattr(evt, attr, default)


class MemoryService:
    """Read/write the per-project CampaignMemory store. Cheap to instantiate."""

    # ------------------------------------------------------------------
    # Write path (called on campaign completion)
    # ------------------------------------------------------------------

    def consolidate_campaign(
        self,
        *,
        project_id: str,
        task_id: Optional[str],
        goal_statement: str,
        outcome: str,
        final_answer: Optional[str],
        events: Optional[List[Any]] = None,
        confidence: str = "hypothesis",
    ) -> Optional[str]:
        """Write one CampaignMemory row distilled from the campaign's event log.

        Never raises — memory is an enhancement, not a critical-path dependency;
        the caller wraps this defensively too.
        """
        try:
            plan_skeleton, dead_ends, key_facts, citations = self._mine_events(events or [])
        except Exception:
            logger.exception("CampaignMemory event mining failed for task %s", task_id)
            plan_skeleton, dead_ends, key_facts, citations = [], [], [], []

        goal = (goal_statement or "").strip()
        session = get_project_session(project_id)
        try:
            row = CampaignMemory(
                project_id=project_id,
                task_id=task_id,
                goal_statement=goal,
                target=goal[:120],
                plan_skeleton=plan_skeleton,
                final_answer=(final_answer or "")[:4000],
                outcome=outcome,
                key_facts=key_facts,
                dead_ends=dead_ends,
                citations=citations,
                confidence=confidence,
            )
            session.add(row)
            session.commit()
            logger.info(
                "CampaignMemory written for task %s: %d steps, %d dead-ends, %d facts",
                task_id, len(plan_skeleton), len(dead_ends), len(key_facts),
            )
            return str(row.id)
        except Exception:
            session.rollback()
            logger.exception("CampaignMemory write failed for task %s", task_id)
            return None
        finally:
            session.close()

    def _mine_events(self, events: List[Any]):
        """Distill (plan_skeleton, dead_ends, key_facts, citations) from events."""
        call_to_tool: Dict[str, str] = {}
        plan_skeleton: List[str] = []
        dead_ends: List[Dict[str, str]] = []
        key_facts: List[str] = []
        citations: List[Any] = []

        for evt in events:
            etype = _event_field(evt, "event_type")
            etype = getattr(etype, "value", etype)  # enum -> str
            payload = _event_field(evt, "payload") or {}

            if etype == _TOOL_INTENT:
                name = payload.get("tool_name")
                call_id = payload.get("call_id")
                if name:
                    if not plan_skeleton or plan_skeleton[-1] != name:
                        plan_skeleton.append(name)
                    if call_id:
                        call_to_tool[call_id] = name

            elif etype == _TOOL_RESULT:
                status = str(payload.get("status") or "").lower()
                out = payload.get("output") or {}
                summary = str(out.get("summary") or "").strip()
                tool = call_to_tool.get(payload.get("call_id"), "tool")
                if "error" in status:
                    dead_ends.append({"action": tool, "reason": summary[:200] or status})
                elif summary:
                    key_facts.append(f"{tool}: {summary[:180]}")
                for key, value in out.items():
                    if any(h in key.lower() for h in _CITATION_KEY_HINTS) and value:
                        citations.append({tool: value})

        # Keep the most recent, capped — a forever loop must not bloat a row.
        return (
            plan_skeleton,
            dead_ends[-_MAX_DEAD_ENDS:],
            key_facts[-_MAX_KEY_FACTS:],
            citations[:_MAX_KEY_FACTS],
        )

    # ------------------------------------------------------------------
    # Read path (called by recall_memory / recall_prior_outcomes tools)
    # ------------------------------------------------------------------

    def recall(
        self,
        *,
        project_id: str,
        query: str,
        limit: int = 5,
        outcome: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Rank prior campaigns by token-overlap × recency-decay × access count."""
        rows = self._load_rows(project_id, outcome=outcome)
        q_tokens = _tokenize(query)
        scored: List[tuple] = []
        for row in rows:
            haystack = " ".join(
                [row.goal_statement or "", row.final_answer or "", " ".join(row.key_facts or [])]
            )
            overlap = len(q_tokens & _tokenize(haystack)) if q_tokens else 0
            recency = self._recency_weight(row.created_at)
            score = overlap * 2.0 + recency + min(row.access_count or 0, 5) * 0.1
            if overlap > 0 or not q_tokens:
                scored.append((score, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [row for _, row in scored[:limit]]
        self._touch(project_id, [row.id for row in top])
        return [self._to_dict(row) for row in top]

    def recall_prior_outcomes(
        self,
        *,
        project_id: str,
        target: str,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Return prior plans + the dead-ends to skip for a related target.

        This is the pruning tool behind the compounding demo beat: before acting
        on a new (related) target, the orchestrator calls this and skips chains a
        prior campaign already proved fruitless.
        """
        rows = self.recall(project_id=project_id, query=target, limit=limit)
        confirmed: List[Dict[str, Any]] = []
        dead_ends: List[Dict[str, Any]] = []
        for row in rows:
            confirmed.append(
                {
                    "campaign": row["goal"][:120],
                    "plan_skeleton": row["plan_skeleton"],
                    "outcome": row["outcome"],
                    "when": row["created_at"],
                    "confidence": row["confidence"],
                }
            )
            for dead in row["dead_ends"]:
                dead_ends.append({**dead, "from_campaign": row["goal"][:80]})

        if not rows:
            summary = f"No prior campaigns found related to '{target}'. Starting cold."
        else:
            summary = (
                f"Recalled {len(rows)} prior campaign(s) related to '{target}': "
                f"{len(dead_ends)} dead-end(s) to skip, "
                f"{len(confirmed)} reusable plan(s). Treat as hypotheses."
            )
        return {
            "status": "success",
            "summary": summary,
            "prior_campaigns": confirmed,
            "dead_ends_to_skip": dead_ends,
            "count": len(rows),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rows(project_id: str, outcome: Optional[str] = None) -> List[CampaignMemory]:
        session = get_project_session(project_id)
        try:
            query = session.query(CampaignMemory)
            if outcome:
                query = query.filter(CampaignMemory.outcome == outcome)
            # Bound the scan; a forever loop accrues many rows but recall only
            # needs the recent/relevant window (older rows decay out of ranking).
            return query.order_by(CampaignMemory.created_at.desc()).limit(200).all()
        finally:
            session.close()

    @staticmethod
    def _recency_weight(created_at: Optional[datetime]) -> float:
        if not created_at:
            return 0.0
        age_days = max(0.0, (datetime.utcnow() - created_at).total_seconds() / 86400.0)
        return math.exp(-age_days / _RECENCY_HALFLIFE_DAYS)

    @staticmethod
    def _touch(project_id: str, row_ids: List[str]) -> None:
        if not row_ids:
            return
        session = get_project_session(project_id)
        try:
            for row in session.query(CampaignMemory).filter(CampaignMemory.id.in_(row_ids)).all():
                row.access_count = (row.access_count or 0) + 1
                row.last_accessed = datetime.utcnow()
            session.commit()
        except Exception:
            session.rollback()
            logger.debug("CampaignMemory touch failed (non-fatal)", exc_info=True)
        finally:
            session.close()

    @staticmethod
    def _to_dict(row: CampaignMemory) -> Dict[str, Any]:
        return {
            "id": str(row.id),
            "goal": row.goal_statement or "",
            "target": row.target or "",
            "plan_skeleton": list(row.plan_skeleton or []),
            "key_facts": list(row.key_facts or []),
            "dead_ends": list(row.dead_ends or []),
            "citations": list(row.citations or []),
            "final_answer": (row.final_answer or "")[:600],
            "outcome": row.outcome or "",
            "confidence": row.confidence or "hypothesis",
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "access_count": row.access_count or 0,
        }
