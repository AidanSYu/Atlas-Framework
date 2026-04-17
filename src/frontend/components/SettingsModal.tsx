'use client';

import React, { useState, useEffect } from 'react';
import { Settings as SettingsIcon, CheckCircle2, AlertCircle, Loader2, Key, Moon, Sun } from 'lucide-react';
import { api, ConfigKeysUpdate } from '@/lib/api';
import type { AtlasTheme } from '@/hooks/useAtlasTheme';
import { toastError, toastSuccess } from '@/stores/toastStore';

interface SettingsModalProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    theme: AtlasTheme;
    onThemeChange: (theme: AtlasTheme) => void;
    onKeysUpdated?: () => void;
}

export default function SettingsModal({ open, onOpenChange, theme, onThemeChange, onKeysUpdated }: SettingsModalProps) {
    const [loading, setLoading] = useState(false);
    const [saving, setSaving] = useState(false);
    const [status, setStatus] = useState({
        openai: false,
        anthropic: false,
        deepseek: false,
        minimax: false,
    });
    const [verified, setVerified] = useState({
        openai: false,
        anthropic: false,
        deepseek: false,
        minimax: false,
    });

    const [keys, setKeys] = useState({
        openai: '',
        anthropic: '',
        deepseek: '',
        minimax: '',
    });

    // Load status and verify keys when modal opens
    useEffect(() => {
        if (!open) return;
        const loadStatus = async () => {
            setLoading(true);
            try {
                const [res, verifyRes] = await Promise.all([
                    api.getApiKeysStatus(),
                    api.verifyApiKeys(),
                ]);
                setStatus({
                    openai: res.has_openai,
                    anthropic: res.has_anthropic,
                    deepseek: res.has_deepseek,
                    minimax: res.has_minimax,
                });
                setVerified({
                    openai: verifyRes.openai,
                    anthropic: verifyRes.anthropic,
                    deepseek: verifyRes.deepseek,
                    minimax: verifyRes.minimax,
                });
            } catch (err) {
                console.error('Failed to load API key status', err);
            } finally {
                setLoading(false);
            }
        };
        loadStatus();
    }, [open]);

    const handleSave = async () => {
        if (!keys.openai && !keys.anthropic && !keys.deepseek && !keys.minimax) {
            onOpenChange(false);
            return;
        }

        setSaving(true);
        try {
            const update: ConfigKeysUpdate = {};
            if (keys.openai) update.OPENAI_API_KEY = keys.openai;
            if (keys.anthropic) update.ANTHROPIC_API_KEY = keys.anthropic;
            if (keys.deepseek) update.DEEPSEEK_API_KEY = keys.deepseek;
            if (keys.minimax) update.MINIMAX_API_KEY = keys.minimax;

            await api.updateApiKeys(update);
            toastSuccess('API keys saved successfully!');

            // Verify keys and update status so "Verified" only shows when key works
            setKeys({ openai: '', anthropic: '', deepseek: '', minimax: '' });
            const [res, verifyRes] = await Promise.all([
                api.getApiKeysStatus(),
                api.verifyApiKeys(),
            ]);
            setStatus({
                openai: res.has_openai,
                anthropic: res.has_anthropic,
                deepseek: res.has_deepseek,
                minimax: res.has_minimax,
            });
            setVerified({
                openai: verifyRes.openai,
                anthropic: verifyRes.anthropic,
                deepseek: verifyRes.deepseek,
                minimax: verifyRes.minimax,
            });
            const anyFailed = (res.has_openai && !verifyRes.openai) || (res.has_anthropic && !verifyRes.anthropic)
                || (res.has_deepseek && !verifyRes.deepseek) || (res.has_minimax && !verifyRes.minimax);
            if (anyFailed) {
                toastError(
                    'Keys saved. Verification failed for some keys (network, timeout, or provider limits can cause this). You can still try using the model.'
                );
            }

            onOpenChange(false);
            onKeysUpdated?.();
        } catch (err) {
            toastError('Failed to save API keys');
            console.error(err);
        } finally {
            setSaving(false);
        }
    };

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm px-4">
            {/* Click outside to close */}
            <div className="absolute inset-0" onClick={() => onOpenChange(false)} />

            <div className="relative w-full max-w-md rounded-xl border border-border bg-card shadow-2xl animate-in zoom-in-95 duration-200">
                <div className="p-6">
                    <div className="flex items-center gap-3 mb-6">
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                            <SettingsIcon className="h-5 w-5" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold tracking-tight text-foreground">Settings</h2>
                            <p className="text-sm text-muted-foreground">Configure AI providers and workspace preferences</p>
                        </div>
                    </div>

                    <div className="rounded-xl border border-border bg-background/60 p-4">
                        <div className="mb-4 flex items-start justify-between gap-3">
                            <div className="flex flex-col gap-1">
                                <h3 className="text-sm font-medium text-foreground">API Providers</h3>
                                <p className="text-xs leading-5 text-muted-foreground">
                                    Add or rotate provider keys for Atlas integrations. Theme changes live in the footer toggle and apply immediately.
                                </p>
                            </div>
                            {loading && (
                                <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-2.5 py-1 text-[10px] font-medium uppercase tracking-[0.18em] text-muted-foreground">
                                    <Loader2 className="h-3 w-3 animate-spin" />
                                    Checking
                                </div>
                            )}
                        </div>

                        <div className="space-y-4">
                            <ApiKeyInput
                                label="OpenAI API Key"
                                value={keys.openai}
                                isConfigured={status.openai}
                                isVerified={verified.openai}
                                onChange={(v) => setKeys(prev => ({ ...prev, openai: v }))}
                                placeholder="sk-..."
                            />
                            <ApiKeyInput
                                label="Anthropic API Key"
                                value={keys.anthropic}
                                isConfigured={status.anthropic}
                                isVerified={verified.anthropic}
                                onChange={(v) => setKeys(prev => ({ ...prev, anthropic: v }))}
                                placeholder="sk-ant-..."
                            />
                            <ApiKeyInput
                                label="DeepSeek API Key"
                                value={keys.deepseek}
                                isConfigured={status.deepseek}
                                isVerified={verified.deepseek}
                                onChange={(v) => setKeys(prev => ({ ...prev, deepseek: v }))}
                                placeholder="sk-..."
                            />
                            <ApiKeyInput
                                label="MiniMax API Key"
                                value={keys.minimax}
                                isConfigured={status.minimax}
                                isVerified={verified.minimax}
                                onChange={(v) => setKeys(prev => ({ ...prev, minimax: v }))}
                                placeholder="..."
                            />
                        </div>
                    </div>

                    <div className="mt-8 flex items-end justify-between gap-4">
                        <FooterThemeToggle theme={theme} onThemeChange={onThemeChange} />

                        <div className="flex items-center gap-3">
                            <button
                                onClick={() => onOpenChange(false)}
                                className="rounded-lg px-4 py-2 text-sm font-medium text-muted-foreground hover:bg-surface hover:text-foreground transition-colors"
                            >
                                Close
                            </button>
                            <button
                                onClick={handleSave}
                                disabled={saving}
                                className="flex items-center justify-center gap-2 rounded-lg bg-primary px-5 py-2 text-sm font-medium text-primary-foreground shadow-sm hover:bg-primary/90 disabled:opacity-50 transition-colors"
                            >
                                {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : 'Save Keys'}
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

function FooterThemeToggle({
    theme,
    onThemeChange,
}: {
    theme: AtlasTheme;
    onThemeChange: (theme: AtlasTheme) => void;
}) {
    const isLight = theme === 'light';

    return (
        <div className="flex min-w-[112px] flex-col items-start gap-2">
            <span className="text-xs font-medium uppercase tracking-[0.24em] text-muted-foreground">
                Theme
            </span>
            <button
                type="button"
                role="switch"
                aria-checked={isLight}
                aria-label={`Switch to ${isLight ? 'dark' : 'light'} mode`}
                onClick={() => onThemeChange(isLight ? 'dark' : 'light')}
                className="relative inline-flex h-8 w-[70px] items-center self-start rounded-full border border-border bg-background transition-colors hover:border-border-strong"
            >
                <span
                    aria-hidden
                    className={`pointer-events-none absolute left-[2px] top-[2px] h-[26px] w-[30px] rounded-full bg-accent shadow-sm transition-transform duration-200 ${
                        isLight ? 'translate-x-[34px]' : 'translate-x-0'
                    }`}
                />
                <span className="relative z-10 grid w-full grid-cols-2 place-items-center">
                    <Moon className={`h-3.5 w-3.5 transition-colors ${
                        isLight ? 'text-muted-foreground' : 'text-accent-foreground'
                    }`} />
                    <Sun className={`h-3.5 w-3.5 transition-colors ${
                        isLight ? 'text-accent-foreground' : 'text-muted-foreground'
                    }`} />
                </span>
            </button>
        </div>
    );
}

function ApiKeyInput({
    label,
    value,
    isConfigured,
    isVerified,
    onChange,
    placeholder
}: {
    label: string;
    value: string;
    isConfigured: boolean;
    isVerified: boolean;
    onChange: (v: string) => void;
    placeholder: string;
}) {
    return (
        <div className="space-y-1.5">
            <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-foreground">{label}</label>
                {isVerified ? (
                    <span className="flex items-center gap-1 text-[10px] uppercase font-bold tracking-wider text-emerald-500">
                        <CheckCircle2 className="h-3 w-3" /> Verified
                    </span>
                ) : isConfigured ? (
                    <span className="flex items-center gap-1 text-[10px] uppercase font-bold tracking-wider text-muted-foreground">
                        <CheckCircle2 className="h-3 w-3" /> Saved
                    </span>
                ) : (
                    <span className="flex items-center gap-1 text-[10px] uppercase font-bold tracking-wider text-muted-foreground">
                        <AlertCircle className="h-3 w-3" /> Not Configured
                    </span>
                )}
            </div>
            <div className="relative">
                <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none text-muted-foreground">
                    <Key className="h-4 w-4" />
                </div>
                <input
                    type="password"
                    value={value}
                    onChange={(e) => onChange(e.target.value)}
                    placeholder={isConfigured ? "Enter new key to replace existing" : placeholder}
                    className="flex h-10 w-full rounded-md border border-border bg-background px-3 py-2 pl-10 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent transition-all"
                />
            </div>
        </div>
    );
}
