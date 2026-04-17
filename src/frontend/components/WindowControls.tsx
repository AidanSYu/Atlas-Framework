'use client';

import React, { useEffect, useState } from 'react';
import { Minus, Square, X, Copy } from 'lucide-react';
import { appWindow } from '@tauri-apps/api/window';

export function WindowControls() {
    const [isMaximized, setIsMaximized] = useState(false);
    const isTauri = typeof window !== 'undefined' && window.__TAURI__ !== undefined;

    useEffect(() => {
        if (!isTauri) return;

        // Check initial state
        appWindow.isMaximized().then(setIsMaximized);

        // Listen for resize events to update the maximize icon
        const unlisten = appWindow.onResized(async () => {
            const maximized = await appWindow.isMaximized();
            setIsMaximized(maximized);
        });

        return () => {
            unlisten.then((f) => f());
        };
    }, [isTauri]);

    if (!isTauri) {
        return null; // Don't render window controls in browser
    }

    return (
        <div className="flex h-full items-center">
            <button
                onClick={() => appWindow.minimize()}
                className="flex h-full w-11 items-center justify-center text-muted-foreground transition-colors hover:bg-white/10"
                title="Minimize"
            >
                <Minus className="h-3.5 w-3.5" />
            </button>

            <button
                onClick={() => appWindow.toggleMaximize()}
                className="flex h-full w-11 items-center justify-center text-muted-foreground transition-colors hover:bg-white/10"
                title={isMaximized ? 'Restore' : 'Maximize'}
            >
                {isMaximized ? (
                    <Copy className="h-3 w-3" /> // A stacked windows icon represents "Restore Down"
                ) : (
                    <Square className="h-3.5 w-3.5" />
                )}
            </button>

            <button
                onClick={() => appWindow.close()}
                className="flex h-full w-11 items-center justify-center text-muted-foreground transition-colors hover:bg-red-500 hover:text-white"
                title="Close"
            >
                <X className="h-4 w-4" />
            </button>
        </div>
    );
}
