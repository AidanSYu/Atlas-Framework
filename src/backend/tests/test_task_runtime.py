"""Coverage for task chat intent and supervisor fallback behavior."""

from __future__ import annotations

import asyncio

from app.core.task_events import EventType
from app.core.task_fsm import TaskState, Trigger
from app.services.task_executor import ExecutorExit, ExecutorResult
from app.services.task_service import TaskService
from app.services.task_supervisor import TaskSupervisor, is_conversational_turn


class _UnavailableLLM:
    async def orchestrate(self, *args, **kwargs):
        raise RuntimeError("missing key")


class _CapturingLog:
    def __init__(self) -> None:
        self.events = []

    def append(self, project_id, task_id, actor, event_type, payload, causal_parents=None):
        self.events.append(
            {
                "project_id": project_id,
                "task_id": task_id,
                "actor": actor,
                "event_type": event_type,
                "payload": payload,
            }
        )

    def list_events(self, project_id, task_id):
        return []


class _StubExecutor:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.briefs = []

    async def run(self, brief, cancel_event=None):
        self.briefs.append(brief)
        return ExecutorResult(exit_reason=ExecutorExit.FINAL_ANSWER, final_answer=self.answer)


def test_short_greeting_is_no_tool_conversational_turn() -> None:
    assert is_conversational_turn("hi") is True
    assert is_conversational_turn("hi", attachments=["C:/tmp/image.png"]) is False


def test_librarian_fallback_does_not_select_first_twelve_tools() -> None:
    supervisor = TaskSupervisor(llm=_UnavailableLLM())

    scoped = asyncio.run(supervisor.scope_manifest("hi"))

    assert scoped.selected == []
    assert "first 12" not in scoped.reasoning


def test_librarian_fallback_uses_small_relevant_scope() -> None:
    supervisor = TaskSupervisor(llm=_UnavailableLLM())

    scoped = asyncio.run(supervisor.scope_manifest("summarize the uploaded papers with citations"))

    assert 1 <= len(scoped.selected) <= 3
    assert "search_literature" in scoped.selected
    assert "first 12" not in scoped.reasoning


def test_task_conductor_lets_executor_generate_greeting_without_real_tools(monkeypatch) -> None:
    log = _CapturingLog()
    executor = _StubExecutor("orchestrator generated answer")
    service = TaskService(log=log, supervisor=TaskSupervisor(llm=_UnavailableLLM()), executor=executor)
    transitions = []
    terminal_outcomes = []

    def fake_transition(project_id, task_id, current, trigger, metadata=None):
        transitions.append((current, trigger, metadata or {}))
        if trigger == Trigger.FINAL_ANSWER_CANDIDATE:
            return TaskState.REVIEWING
        if trigger == Trigger.REVIEW_APPROVE:
            return TaskState.COMPLETED
        return current

    monkeypatch.setattr(service, "_transition", fake_transition)
    monkeypatch.setattr(
        service,
        "_set_terminal",
        lambda project_id, task_id, outcome: terminal_outcomes.append(outcome),
    )

    asyncio.run(service._run_planning_and_execute("project-1", "task-1", "hi"))

    event_types = [event["event_type"] for event in log.events]
    assert event_types == [
        EventType.MANIFEST_SCOPED,
        EventType.SUPERVISOR_BRIEF,
        EventType.FINAL_ANSWER,
        EventType.SUPERVISOR_REVIEW,
    ]
    assert log.events[0]["payload"]["selected_tools"] == []
    assert executor.briefs[0].active_manifest == []
    assert log.events[2]["payload"]["answer"] == "orchestrator generated answer"
    assert any(transition[1] == Trigger.BRIEF_READY for transition in transitions)
    assert any(transition[1] == Trigger.FINAL_ANSWER_CANDIDATE for transition in transitions)
    assert transitions[-1][1] == Trigger.REVIEW_APPROVE
    assert terminal_outcomes == ["completed"]
