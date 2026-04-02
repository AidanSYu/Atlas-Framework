'use client';

import React, { useRef, useEffect, useCallback } from 'react';
import { Send, Loader2, Paperclip, Beaker, X, BookOpen, Brain, Network, MessageSquare } from 'lucide-react';
import type { SpectrumUploadResponse } from '@/lib/api';
import type { ChatMode } from '@/hooks/useRunManager';

// Re-export for consumers that import from here
export type { ChatMode };

// ---------------------------------------------------------------------------
// Mode metadata
// ---------------------------------------------------------------------------

const MODE_META: Record<ChatMode, { label: string; icon: typeof BookOpen; color: string; bg: string; border: string }> = {
  librarian: { label: 'Librarian', icon: BookOpen, color: 'text-primary', bg: 'bg-primary/10', border: 'border-primary/20' },
  cortex: { label: 'Cortex', icon: Brain, color: 'text-accent', bg: 'bg-accent/10', border: 'border-accent/20' },
  moe: { label: 'MoE', icon: Network, color: 'text-blue-500', bg: 'bg-blue-500/10', border: 'border-blue-500/20' },
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CommandSurfaceProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onCancel: () => void;
  isLoading: boolean;
  disabled: boolean;
  chatMode: ChatMode;
  streamingText?: string;
  onStopWithPartial?: (partialText: string) => void;

}

// ---------------------------------------------------------------------------
// Placeholder text per mode
// ---------------------------------------------------------------------------

const PLACEHOLDERS: Record<ChatMode, string> = {
  librarian: 'Ask about your documents...',
  cortex: 'Deep research question...',
  moe: 'Complex synthesis query...',
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function CommandSurface({
  value,
  onChange,
  onSubmit,
  onCancel,
  isLoading,
  disabled,
  chatMode,
  streamingText = '',
  onStopWithPartial,

}: CommandSurfaceProps) {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 140)}px`;
    }
  }, [value]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  }, [onSubmit]);

  const handleStop = useCallback(() => {
    if (onStopWithPartial) {
      onStopWithPartial(streamingText);
    } else {
      onCancel();
    }
  }, [onCancel, onStopWithPartial, streamingText]);

  const placeholder = PLACEHOLDERS[chatMode];



  return (
    <div className="shrink-0 border-t border-border/50 bg-card/80 backdrop-blur-sm">
      {/* Input area */}
      <div className="relative mx-auto max-w-3xl px-4 py-3">

        <div className="flex items-end gap-2 rounded-2xl border border-border/40 bg-surface/30 transition-all focus-within:border-border/80 focus-within:bg-surface/50">
          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            className={`flex-1 resize-none bg-transparent py-3.5 text-sm text-foreground outline-none placeholder:text-muted-foreground/40 pl-4 pr-2`}
            rows={1}
            disabled={isLoading || disabled}
          />

          <div className="flex items-center gap-1 pr-2 pb-2">
            {isLoading ? (
              <button
                onClick={handleStop}
                className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-destructive bg-destructive/10 hover:bg-destructive/20 transition-all"
              >
                Stop
              </button>
            ) : (
              <button
                onClick={onSubmit}
                disabled={!value.trim()}
                className={`rounded-xl p-2 transition-all ${!value.trim()
                  ? 'text-muted-foreground/30'
                  : 'bg-accent text-white hover:bg-accent/90 shadow-sm shadow-accent/20'
                }`}
              >
                <Send className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        {/* Hint line */}
        <div className="flex items-center justify-between mt-1.5 px-1">
          <span className="text-[10px] text-muted-foreground/40">
            <kbd className="font-mono">Shift+Enter</kbd> for new line
          </span>
          {!isLoading && (
            <span className="text-[10px] text-muted-foreground/40">
              <kbd className="font-mono">Ctrl+K</kbd> ask from anywhere
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
