'use client';

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Upload, File, Trash2, Loader2 } from 'lucide-react';
import { api, FileInfo } from '@/lib/api';
import { toastError } from '@/stores/toastStore';

interface FileSidebarProps {
  onFileSelect: (docId: string, filename: string) => void;
  selectedDocId: string | null;
  projectId?: string;
  onIngestionComplete?: () => void;
  onFileDeleted?: (docId: string) => void;
}

export default function FileSidebar({ onFileSelect, selectedDocId, projectId, onIngestionComplete, onFileDeleted }: FileSidebarProps) {
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(true);
  const prevProcessingRef = useRef(false);

  const loadFiles = useCallback(async (silent = false) => {
    if (!projectId) return;
    try {
      if (!silent) setLoading(true);
      const fileList = await api.listFiles(projectId);
      setFiles(fileList);
    } catch (error) {
      console.error('Failed to load files:', error);
    } finally {
      if (!silent) setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    loadFiles();
  }, [loadFiles]);

  // Poll for status updates while any file is processing
  const hasProcessing = files.some(f => f.status === 'processing');

  useEffect(() => {
    if (!hasProcessing) {
      // If we were processing before and now we're not, ingestion just finished
      if (prevProcessingRef.current && onIngestionComplete) {
        onIngestionComplete();
      }
      prevProcessingRef.current = false;
      return;
    }

    prevProcessingRef.current = true;

    const interval = window.setInterval(() => {
      loadFiles(true); // silent reload (no loading spinner)
    }, 3000);

    return () => window.clearInterval(interval);
  }, [hasProcessing, loadFiles, onIngestionComplete]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file || !projectId) return;

    setUploading(true);
    try {
      await api.uploadFile(file, projectId);
      await loadFiles();
    } catch (error) {
      toastError((error as Error).message);
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, fileId: string) => {
    e.stopPropagation();
    if (!projectId) return;
    if (!confirm('Delete this file?')) return;
    try {
      await api.deleteFile(fileId, projectId);
      onFileDeleted?.(fileId);
      setFiles(prev => prev.filter(f => f.doc_id !== fileId));
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="h-full flex flex-col bg-card">
      {/* File List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-2 space-y-1">
        {loading ? (
          <div className="flex justify-center py-8">
            <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
          </div>
        ) : files.length === 0 ? (
          <div className="text-center py-8 px-4 text-muted-foreground text-xs">
            <Upload className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No files uploaded.</p>
          </div>
        ) : (
          files.map((file) => (
            <div
              key={file.doc_id}
              onClick={() => onFileSelect(file.doc_id, file.filename)}
              className={`group flex items-center justify-between p-2 rounded-md cursor-pointer transition-all border border-transparent ${selectedDocId === file.doc_id
                  ? 'bg-accent border-border text-foreground'
                  : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
                }`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <File className="w-4 h-4 shrink-0 opacity-70" />
                <div className="min-w-0">
                  <p className="text-xs font-medium truncate">{file.filename}</p>
                  <p className="text-[10px] opacity-60 flex items-center gap-1">
                    {file.status === 'completed' ? (
                      <span className="text-emerald-500">ready</span>
                    ) : file.status === 'processing' ? (
                      <span className="text-yellow-500 flex items-center gap-1">
                        <Loader2 className="w-2.5 h-2.5 animate-spin" />
                        processing
                      </span>
                    ) : file.status === 'indexed' ? (
                      <span className="text-emerald-500 capitalize">{file.status}</span>
                    ) : (
                      <span className="text-yellow-500 capitalize">{file.status}</span>
                    )}
                    <span>•</span>
                    <span>{((file.size_bytes || 0) / 1024).toFixed(0)} KB</span>
                  </p>
                </div>
              </div>

              <button
                onClick={(e) => handleDelete(e, file.doc_id)}
                className="p-1 opacity-0 group-hover:opacity-100 hover:text-destructive transition-opacity"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          ))
        )}
      </div>

      {/* Upload Area */}
      <div className="p-3 border-t border-border bg-card">
        <label className={`flex items-center justify-center gap-2 w-full p-2 border border-dashed border-border rounded-lg cursor-pointer hover:bg-accent/50 transition-colors ${uploading ? 'opacity-50 pointer-events-none' : ''}`}>
          <input type="file" className="hidden" onChange={handleFileUpload} accept=".pdf,.txt,.docx,.doc" disabled={uploading} />
          {uploading ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Upload className="w-4 h-4 text-muted-foreground" />
          )}
          <span className="text-xs font-medium text-muted-foreground">
            {uploading ? 'Processing...' : 'Upload File'}
          </span>
        </label>
      </div>
    </div>
  );
}
