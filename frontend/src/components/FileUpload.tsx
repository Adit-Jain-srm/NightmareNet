"use client";

import { useState, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, CheckCircle2, X, Loader2, AlertCircle, Zap, Shield } from "lucide-react";
import { uploadTextFile, type UploadResponse } from "@/lib/api";

export default function FileUpload() {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await uploadTextFile(file);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <section id="upload" className="relative py-28 px-6">
      <div className="max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="text-[10px] font-mono text-success uppercase tracking-[0.2em] mb-3 block">
            Data Input
          </span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            Upload <span className="text-gradient-neural">Data</span>
          </h2>
          <p className="text-text-dim max-w-md mx-auto text-sm">
            Feed your text data into the distortion pipeline. Supports .txt, .csv, and .json.
          </p>
        </motion.div>

        {/* Drop zone */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`relative glass-card p-12 text-center cursor-pointer transition-all duration-300 ${
              dragging
                ? "!border-neural/40 box-glow-neural scale-[1.01]"
                : "hover:!border-white/10"
            }`}
          >
            <input
              ref={inputRef}
              type="file"
              accept=".txt,.csv,.json"
              onChange={onSelect}
              className="hidden"
            />

            {loading ? (
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="w-10 h-10 text-neural animate-spin" />
                <p className="text-sm text-text-dim">Processing file...</p>
              </div>
            ) : (
              <>
                <div className="w-16 h-16 rounded-2xl bg-neural/5 border border-neural/10 flex items-center justify-center mx-auto mb-4">
                  <Upload className={`w-7 h-7 transition-colors ${dragging ? "text-neural" : "text-slate-400"}`} />
                </div>
                <p className="text-sm text-text-dim mb-1">
                  {dragging ? "Drop your file here" : "Drag & drop or click to upload"}
                </p>
                <p className="text-xs text-slate-400">.txt, .csv, .json • Max 5MB</p>
              </>
            )}

            {/* Animated border on drag */}
            {dragging && (
              <div className="absolute inset-0 rounded-[1rem] border-gradient-animated pointer-events-none" />
            )}
          </div>
        </motion.div>

        {/* Results */}
        <AnimatePresence mode="wait">
          {error && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-6 glass-card p-4 !border-nightmare/20"
            >
              <div className="flex items-start gap-3">
                <AlertCircle className="w-4 h-4 text-nightmare shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-nightmare font-medium">Upload Failed</p>
                  <p className="text-xs text-slate-400 mt-1">{error}</p>
                </div>
                <button onClick={() => setError(null)} className="ml-auto text-slate-400 hover:text-text cursor-pointer">
                  <X className="w-4 h-4" />
                </button>
              </div>
            </motion.div>
          )}

          {result && (
            <motion.div
              key="result"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="mt-6 glass-card overflow-hidden"
            >
              {/* File info header */}
              <div className="flex items-center gap-3 p-4 border-b border-white/[0.04]">
                <div className="w-10 h-10 rounded-xl bg-success/5 border border-success/10 flex items-center justify-center">
                  <CheckCircle2 className="w-5 h-5 text-success" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-text truncate">{result.filename}</p>
                  <p className="text-[10px] font-mono text-slate-400">{result.file_type} file</p>
                </div>
                <button onClick={() => setResult(null)} className="text-slate-400 hover:text-text cursor-pointer">
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 divide-x divide-white/[0.04] border-b border-white/[0.04]">
                {[
                  { label: "Characters", value: result.char_count.toLocaleString() },
                  { label: "Words", value: result.word_count.toLocaleString() },
                  { label: "Lines", value: result.line_count.toLocaleString() },
                ].map((s) => (
                  <div key={s.label} className="text-center p-3">
                    <p className="text-sm font-bold font-mono text-neural">{s.value}</p>
                    <p className="text-[9px] text-slate-400 uppercase">{s.label}</p>
                  </div>
                ))}
              </div>

              <p className="text-xs text-slate-400 px-4 pt-2 border-b border-white/[0.04] pb-3">
                For an E2E training run, open{" "}
                <a href="#pipeline" className="text-neural hover:underline cursor-pointer">
                  Pipeline
                </a>
                , choose <span className="text-text">Paste text</span>, and paste this file&apos;s
                contents (the API uses <span className="font-mono text-[10px]">source_type: &quot;text&quot;</span>
                ).
              </p>

              {/* Preview */}
              <div className="p-4">
                <div className="terminal">
                  <div className="terminal-header">
                    <div className="terminal-dot bg-nightmare/60" />
                    <div className="terminal-dot bg-warning/60" />
                    <div className="terminal-dot bg-success/60" />
                    <span className="text-[10px] font-mono text-slate-400 ml-2">preview</span>
                  </div>
                  <div className="p-4 max-h-48 overflow-y-auto">
                    <pre className="text-xs text-text-dim whitespace-pre-wrap leading-relaxed">
                      {result.preview}
                    </pre>
                  </div>
                </div>
              </div>

              {/* Action buttons */}
              <div className="flex items-center gap-3 p-4 border-t border-white/[0.04]">
                <button
                  onClick={() => {
                    const textToCopy = result.text_content.slice(0, 500);
                    navigator.clipboard.writeText(textToCopy);
                    const playgroundSection = document.getElementById("playground");
                    playgroundSection?.scrollIntoView({ behavior: "smooth" });
                  }}
                  className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-medium bg-dream/10 border border-dream/15 text-dream hover:bg-dream/15 transition-colors cursor-pointer"
                >
                  <Zap className="w-3.5 h-3.5" />
                  Use in Playground
                </button>
                <button
                  onClick={() => {
                    const textToCopy = result.text_content.slice(0, 500);
                    navigator.clipboard.writeText(textToCopy);
                    const resilienceSection = document.getElementById("resilience");
                    resilienceSection?.scrollIntoView({ behavior: "smooth" });
                  }}
                  className="flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-medium bg-neural/10 border border-neural/15 text-neural hover:bg-neural/15 transition-colors cursor-pointer"
                >
                  <Shield className="w-3.5 h-3.5" />
                  Use in Resilience Lab
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
