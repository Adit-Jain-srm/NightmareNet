"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Brain, Layers, Network, Cpu, ChevronRight, Zap, Database, GitBranch } from "lucide-react";

interface LayerInfo {
  name: string;
  type: string;
  params: string;
  color: string;
  desc: string;
}

const MODEL_DATA: Record<string, { layers: LayerInfo[]; totalParams: string; modelName: string; modelDesc: string }> = {
  causal_lm: {
    layers: [
      { name: "Token Embeddings", type: "Embedding", params: "38.6M", color: "neural", desc: "Maps input token IDs to dense 768-dimensional vectors. Vocabulary size: 50,257 tokens covering BPE-encoded English text." },
      { name: "Position Embeddings", type: "Embedding", params: "98.3K", color: "neural", desc: "Learnable positional encodings for up to 1,024 token positions, enabling the model to understand token order." },
      { name: "Transformer Block ×12", type: "Attention + FFN", params: "85.1M", color: "dream", desc: "12 stacked transformer blocks, each containing multi-head self-attention (12 heads) + feed-forward network with GELU activation. Causal masking ensures autoregressive generation." },
      { name: "Layer Norm", type: "Normalization", params: "1.5K", color: "success", desc: "Final layer normalization applied to the output of the last transformer block, stabilizing hidden state magnitudes." },
      { name: "LM Head", type: "Linear", params: "38.6M", color: "warning", desc: "Projects hidden states back to vocabulary logits for next-token prediction. Weight-tied with token embeddings." },
    ],
    totalParams: "124.4M",
    modelName: "GPT-2",
    modelDesc: "AutoModelForCausalLM • 124M params • wikitext-2",
  },
  masked_lm: {
    layers: [
      { name: "Token Embeddings", type: "Embedding", params: "23.4M", color: "neural", desc: "Maps token IDs to 768-dim vectors. BERT uses WordPiece tokenization with a 30,522-token vocabulary." },
      { name: "Position Embeddings", type: "Embedding", params: "393K", color: "neural", desc: "Learnable positional encodings for up to 512 positions. Combined with token type embeddings for segment awareness." },
      { name: "Token Type Embeddings", type: "Embedding", params: "1.5K", color: "success", desc: "Segment embeddings distinguishing sentence A from sentence B, critical for NSP and sentence-pair tasks." },
      { name: "Encoder Block ×12", type: "Bi-Attention + FFN", params: "85.1M", color: "dream", desc: "12 encoder layers with bidirectional self-attention (12 heads) enabling full context on both sides. No causal mask — all tokens attend to all tokens." },
      { name: "MLM Head", type: "Linear + LN + GELU", params: "23.5M", color: "warning", desc: "Predicts masked tokens: dense → GELU → LayerNorm → projection to vocabulary logits. Trained to reconstruct randomly masked inputs." },
    ],
    totalParams: "109.5M",
    modelName: "BERT-base",
    modelDesc: "AutoModelForMaskedLM • 110M params • wikipedia",
  },
  seq_class: {
    layers: [
      { name: "Token Embeddings", type: "Embedding", params: "14.5M", color: "neural", desc: "Maps token IDs to 768-dim vectors. DistilBERT preserves the same vocabulary as BERT-base while halving the depth." },
      { name: "Position Embeddings", type: "Embedding", params: "393K", color: "neural", desc: "Sinusoidal positional encodings for up to 512 positions. Distilled from BERT's learned embeddings." },
      { name: "Encoder Block ×6", type: "Bi-Attention + FFN", params: "42.5M", color: "dream", desc: "6 distilled transformer layers maintaining 97% of BERT's performance. Trained via knowledge distillation from a 12-layer teacher." },
      { name: "Pre-classifier", type: "Linear + ReLU", params: "590K", color: "success", desc: "Dense projection mapping the [CLS] token representation to an intermediate space before the final classification." },
      { name: "Classifier Head", type: "Linear", params: "1.5K", color: "warning", desc: "Maps to num_labels logits. Dropout (0.2) applied before classification for regularization during fine-tuning." },
    ],
    totalParams: "66.4M",
    modelName: "DistilBERT",
    modelDesc: "AutoModelForSequenceClassification • 66M params • sst-2",
  },
};

const MODEL_TYPES = [
  { key: "causal_lm", label: "Causal LM", icon: Brain },
  { key: "masked_lm", label: "Masked LM", icon: Network },
  { key: "seq_class", label: "Classification", icon: Layers },
];

const FEATURES = [
  { icon: Zap, label: "Mixed Precision", desc: "AMP + gradient checkpointing for 2× speed" },
  { icon: Database, label: "Streaming Data", desc: "Process datasets larger than memory" },
  { icon: GitBranch, label: "Distributed", desc: "Multi-GPU via HuggingFace Accelerate" },
  { icon: Cpu, label: "Multi-Arch", desc: "GPT-2, BERT, DistilBERT, and more" },
];

export default function ModelViewer() {
  const [selectedType, setSelectedType] = useState("causal_lm");
  const [expandedLayer, setExpandedLayer] = useState<number | null>(null);
  const activeModel = MODEL_DATA[selectedType];

  return (
    <section id="model-viewer" className="relative py-28 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="text-[10px] font-mono text-dream uppercase tracking-[0.2em] mb-3 block">
            Architecture
          </span>
          <h2 className="text-3xl md:text-5xl font-black tracking-tight mb-4">
            Model <span className="text-gradient-dream">Explorer</span>
          </h2>
          <p className="text-text-dim max-w-lg mx-auto text-sm">
            Inspect the transformer architecture that NightmareNet trains, distorts, and compresses.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Left: Model type selector + layer stack */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
          >
            {/* Model type */}
            <div className="flex gap-2 mb-6">
              {MODEL_TYPES.map((t) => (
                <button
                  key={t.key}
                  onClick={() => { setSelectedType(t.key); setExpandedLayer(null); }}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-all cursor-pointer ${
                    selectedType === t.key
                      ? "text-text glass-card !border-dream/20 box-glow-dream"
                      : "text-slate-400 hover:text-text-dim bg-white/[0.01] border border-white/[0.04] rounded-xl"
                  }`}
                >
                  <t.icon className="w-3.5 h-3.5" />
                  {t.label}
                </button>
              ))}
            </div>

            {/* Layer stack */}
            <div className="space-y-2">
              {activeModel.layers.map((layer, i) => (
                <motion.div
                  key={layer.name}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.06 }}
                >
                  <button
                    onClick={() => setExpandedLayer(expandedLayer === i ? null : i)}
                    className={`w-full text-left glass-card p-4 cursor-pointer group ${
                      expandedLayer === i ? `!border-${layer.color}/20 box-glow-${layer.color}` : ""
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      {/* Layer indicator */}
                      <div className={`w-1 h-8 rounded-full bg-${layer.color}/50`} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <span className={`text-sm font-semibold text-${layer.color}`}>
                            {layer.name}
                          </span>
                          <ChevronRight
                            className={`w-4 h-4 text-slate-400 transition-transform duration-200 ${
                              expandedLayer === i ? "rotate-90" : ""
                            }`}
                          />
                        </div>
                        <div className="flex items-center gap-3 mt-0.5">
                          <span className="text-[10px] font-mono text-slate-400">{layer.type}</span>
                          <span className="text-[10px] font-mono text-slate-400/60">•</span>
                          <span className={`text-[10px] font-mono text-${layer.color}/70`}>
                            {layer.params} params
                          </span>
                        </div>
                      </div>
                    </div>

                    <AnimatePresence>
                      {expandedLayer === i && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden"
                        >
                          <div className="pt-3 ml-4 border-t border-white/[0.04] mt-3">
                            <p className="text-xs text-slate-400 leading-relaxed">
                              {layer.desc}
                            </p>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </button>
                </motion.div>
              ))}
            </div>

            {/* Total params */}
            <div className="mt-4 flex items-center justify-between px-4 py-3 rounded-xl bg-white/[0.02] border border-white/[0.04]">
              <span className="text-xs text-slate-400 font-mono">Total Parameters</span>
              <span className="text-sm font-bold text-gradient-neural font-mono">{activeModel.totalParams}</span>
            </div>
          </motion.div>

          {/* Right: Feature cards */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="space-y-4"
          >
            <h3 className="text-sm font-semibold text-text-dim mb-4">Training Capabilities</h3>
            {FEATURES.map((f, i) => (
              <motion.div
                key={f.label}
                initial={{ opacity: 0, y: 15 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: 0.1 + i * 0.06 }}
                className="glass-card p-5 flex items-start gap-4"
              >
                <div className="shrink-0 w-10 h-10 rounded-xl bg-neural/5 border border-neural/10 flex items-center justify-center">
                  <f.icon className="w-5 h-5 text-neural" />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-text mb-0.5">{f.label}</h4>
                  <p className="text-xs text-slate-400 leading-relaxed">{f.desc}</p>
                </div>
              </motion.div>
            ))}

            {/* Model info card */}
            <div className="glass-card p-5 mt-6">
              <h4 className="text-xs font-mono text-slate-400 uppercase tracking-wider mb-3">Default Model</h4>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-dream/5 border border-dream/10 flex items-center justify-center">
                  <Brain className="w-5 h-5 text-dream" />
                </div>
                <div>
                  <p className="text-sm font-bold text-text">{activeModel.modelName}</p>
                  <p className="text-[10px] font-mono text-slate-400">{activeModel.modelDesc}</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
