'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Send, FlaskConical, Beaker, MessageSquare } from 'lucide-react';
import { api, getApiBase } from '@/lib/api';
import { streamSSE, type NormalizedEvent } from '@/lib/stream-adapter';

interface CoordinatorChatProps {
  sessionId: string;
  projectId: string;
  onCoordinatorComplete?: (goals: string[]) => void;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isThinking?: boolean;
}

export function CoordinatorChat({ sessionId, projectId, onCoordinatorComplete }: CoordinatorChatProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bootstrapTriggered = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = useCallback(async (contentToSubmit: string) => {
    if (isRunning) return;

    const trimmed = contentToSubmit.trim();
    if (trimmed) {
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'user', content: trimmed }]);
    }
    
    setInput('');
    setIsRunning(true);
    abortRef.current = new AbortController();

    try {
      let finalResult = null;
      let wasCancelled = false;
      const url = `${getApiBase()}/api/discovery/${sessionId}/coordinator/chat`;
      const body = { message: trimmed || null, project_id: projectId };

      await streamSSE(url, body, (event: NormalizedEvent) => {
        if (event.type === 'coordinator_thinking') {
          setMessages(prev => {
            const last = prev[prev.length - 1];
            if (last?.role === 'assistant' && last?.isThinking) {
              const updated = [...prev];
              updated[updated.length - 1] = { ...last, content: last.content + '\n' + event.content };
              return updated;
            }
            return [...prev, { id: crypto.randomUUID(), role: 'assistant', content: event.content, isThinking: true }];
          });
        } else if (event.type === 'coordinator_question') {
          let questionContent = '';
          if (event.context) questionContent += `*${event.context}*\n\n`;
          questionContent += event.question;
          if (event.goalsSoFar?.length > 0) {
            questionContent += `\n\n**Goals so far:** ${event.goalsSoFar.map((g: string) => `\`${g}\``).join(', ')}`;
          }
          setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'assistant', content: questionContent }]);
        } else if (event.type === 'coordinator_complete') {
          finalResult = event;
          let completionMsg = `**Session configured!** ${event.summary}\n\n`;
          if (event.extractedGoals && event.extractedGoals.length > 0) {
            completionMsg += `**Extracted Goals:**\n${event.extractedGoals.map((g: string) => `- ${g}`).join('\n')}\n\n`;
          }
          setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'assistant', content: completionMsg }]);
          onCoordinatorComplete?.(event.extractedGoals || []);
        } else if (event.type === 'error') {
          setMessages(prev => [...prev, { id: crypto.randomUUID(), role: 'assistant', content: `Error: ${event.message}` }]);
        }
      }, { signal: abortRef.current.signal, timeout: 300_000 });

    } catch (err: any) {
      if (err?.name !== 'AbortError') {
        const errMsg = err?.message || 'Unknown error';
        setMessages(prev => [...prev, {
          id: crypto.randomUUID(), role: 'assistant',
          content: `Coordinator hit an error. Try clicking send again to resume. Details: ${errMsg}`
        }]);
      }
    } finally {
      setIsRunning(false);
    }
  }, [sessionId, projectId, isRunning, onCoordinatorComplete]);

  // Bootstrap
  useEffect(() => {
    if (!bootstrapTriggered.current) {
      bootstrapTriggered.current = true;
      handleSubmit('');
    }
  }, [handleSubmit]);

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm ${
              msg.role === 'user' 
                ? 'bg-emerald-600 text-white rounded-br-sm' 
                : msg.isThinking
                  ? 'bg-emerald-500/10 text-emerald-600 font-mono text-xs rounded-bl-sm border border-emerald-500/20 whitespace-pre-wrap'
                  : 'bg-card border border-border text-foreground rounded-bl-sm whitespace-pre-wrap'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
        <div ref={scrollRef} />
      </div>
      <div className="shrink-0 border-t border-border p-3">
        <div className="relative flex items-center rounded-xl border border-border bg-surface px-3 py-2 shadow-sm focus-within:border-emerald-500/50">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSubmit(input); }}
            placeholder="Type your answer..."
            disabled={isRunning}
            className="flex-1 bg-transparent px-2 text-sm outline-none disabled:opacity-50"
          />
          <button
            onClick={() => handleSubmit(input)}
            disabled={isRunning || !input.trim()}
            className="ml-2 rounded-lg bg-emerald-500 p-1.5 text-white transition-colors hover:bg-emerald-600 disabled:opacity-50"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
