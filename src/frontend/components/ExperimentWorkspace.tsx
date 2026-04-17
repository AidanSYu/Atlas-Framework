'use client';

import React from 'react';
import { FlaskConical, Plus, Loader2 } from 'lucide-react';

import { TaskChat } from '@/components/TaskChat';
import type { TaskInfo } from '@/lib/api';

interface ExperimentWorkspaceProps {
  projectId: string;
  activeTaskId: string | null;
  onNewTask: () => void;
  onTaskCreated?: (task: TaskInfo) => void;
  creating?: boolean;
}

/**
 * Task-mode canvas. The task list + "new task" affordances live in ProjectSidebar;
 * this component renders only the active task chat (or an empty state).
 */
export function ExperimentWorkspace({
  projectId,
  activeTaskId,
  onNewTask,
  onTaskCreated,
  creating,
}: ExperimentWorkspaceProps) {
  if (activeTaskId) {
    return (
      <div className="h-full min-h-0">
        <TaskChat
          key={activeTaskId}
          projectId={projectId}
          taskId={activeTaskId}
          onTaskCreated={onTaskCreated}
        />
      </div>
    );
  }

  return <EmptyState onNewTask={onNewTask} creating={creating ?? false} />;
}

function EmptyState({ onNewTask, creating }: { onNewTask: () => void; creating: boolean }) {
  return (
    <div className="flex h-full flex-col items-center justify-center px-8 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl border border-accent/20 bg-accent/5">
        <FlaskConical className="h-6 w-6 text-accent" />
      </div>
      <h2 className="mt-5 text-xl font-semibold text-foreground">Start a new task</h2>
      <p className="mt-2 max-w-md text-[13px] text-muted-foreground leading-relaxed">
        DeepSeek scopes the tools and writes the goal brief. Nemotron runs the tools dynamically,
        observing results and deciding what to call next — just like Claude Code, but for research.
      </p>
      <button
        onClick={onNewTask}
        disabled={creating}
        className="mt-6 inline-flex items-center gap-2 rounded-xl bg-accent px-4 py-2.5 text-[13px] font-medium text-white hover:bg-accent/90 disabled:opacity-50"
      >
        {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
        New Task
      </button>
    </div>
  );
}
