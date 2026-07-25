"use client";

import { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";
import { ChevronDown, ArrowRight, Zap, LayoutDashboard } from "lucide-react";
import Logo from "./Logo";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";

gsap.registerPlugin(useGSAP);

const WORDS = ["Dream", "Nightmare", "Compress", "Evolve"];

function useTypewriter(words: string[], speed = 100, pause = 2000) {
  const [text, setText] = useState("");
  const [wordIdx, setWordIdx] = useState(0);
  const [charIdx, setCharIdx] = useState(0);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    const word = words[wordIdx];
    const timeout = deleting ? speed / 2 : speed;

    const timer = setTimeout(() => {
      if (!deleting) {
        setText(word.slice(0, charIdx + 1));
        if (charIdx + 1 === word.length) {
          setTimeout(() => setDeleting(true), pause);
        } else {
          setCharIdx(charIdx + 1);
        }
      } else {
        setText(word.slice(0, charIdx));
        if (charIdx === 0) {
          setDeleting(false);
          setWordIdx((wordIdx + 1) % words.length);
        } else {
          setCharIdx(charIdx - 1);
        }
      }
    }, timeout);

    return () => clearTimeout(timer);
  }, [charIdx, deleting, wordIdx, words, speed, pause]);

  return text;
}

const phases = [
  { label: "Wake", color: "text-success", bg: "bg-success/10", border: "border-success/20" },
  { label: "Dream", color: "text-dream", bg: "bg-dream/10", border: "border-dream/20" },
  { label: "Nightmare", color: "text-nightmare", bg: "bg-nightmare/10", border: "border-nightmare/20" },
  { label: "Compress", color: "text-neural", bg: "bg-neural/10", border: "border-neural/20" },
];

const stats = [
  { value: "250+", label: "Tests Passing" },
  { value: "4", label: "Training Phases" },
  { value: "7", label: "API Endpoints" },
  { value: "< 50ms", label: "Avg Latency" },
];

export default function Hero() {
  const typedWord = useTypewriter(WORDS, 80, 2500);
  const containerRef = useRef<HTMLDivElement>(null);

  useGSAP(() => {
    gsap.to(".hero-orb-1", {
      x: 30, y: -20, duration: 8, ease: "sine.inOut", repeat: -1, yoyo: true,
    });
    gsap.to(".hero-orb-2", {
      x: -25, y: 15, duration: 10, ease: "sine.inOut", repeat: -1, yoyo: true,
    });
    gsap.to(".hero-orb-3", {
      x: 15, y: 25, duration: 12, ease: "sine.inOut", repeat: -1, yoyo: true,
    });
  }, { scope: containerRef });

  return (
    <section
      ref={containerRef}
      aria-labelledby="hero-heading"
      className="relative min-h-screen flex flex-col items-center justify-center px-6 overflow-hidden"
    >
      {/* Ambient gradient orbs */}
      <div className="hero-orb-1 absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-dream/[0.04] rounded-full blur-[120px] pointer-events-none" />
      <div className="hero-orb-2 absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-nightmare/[0.03] rounded-full blur-[120px] pointer-events-none" />
      <div className="hero-orb-3 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-neural/[0.02] rounded-full blur-[150px] pointer-events-none" />

      {/* Logo with glow */}
      <motion.div
        initial={{ scale: 0, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 mb-8"
      >
        <div className="relative">
          <Logo size="xl" showText={false} animated={true} />
          <div className="absolute inset-0 bg-neural/15 rounded-full blur-xl animate-pulse-glow" />
        </div>
      </motion.div>

      {/* Title */}
      <motion.h1
        id="hero-heading"
        initial={{ y: 40, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 text-5xl md:text-7xl lg:text-8xl font-black text-center tracking-[-0.04em] leading-[0.9]"
      >
        <span className="text-gradient-dream">Nightmare</span>
        <span className="text-text">Net</span>
      </motion.h1>

      {/* Subtitle with typewriter */}
      <motion.div
        initial={{ y: 30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.7, delay: 0.35, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 mt-6 text-center"
      >
        <p className="text-lg md:text-xl text-text-dim max-w-2xl leading-relaxed">
          Autonomous AI Self-Improvement through
        </p>
        <p className="text-2xl md:text-3xl font-bold mt-2 h-10" aria-hidden="true">
          <span className="text-gradient-neural">{typedWord}</span>
          <span className="text-neural animate-cursor-blink ml-0.5">|</span>
        </p>
        <span className="sr-only">Dream, Nightmare, Compress, and Evolve.</span>
        <p className="text-sm text-slate-400 mt-3 max-w-lg mx-auto">
          Force neural networks to learn invariant structures — not memorize patterns.
        </p>
      </motion.div>

      {/* CTA Buttons */}
      <motion.div
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.55 }}
        className="relative z-10 mt-10 flex flex-wrap items-center justify-center gap-4"
      >
        <a href="/dashboard" className="btn-primary">
          <LayoutDashboard aria-hidden="true" className="w-4 h-4 relative z-10" />
          <span className="relative z-10">Launch Dashboard</span>
        </a>
        <a href="#playground" className="btn-ghost">
          <Zap aria-hidden="true" className="w-4 h-4" />
          Try Playground
        </a>
        <a href="#architecture" className="btn-ghost">
          How It Works
          <ArrowRight aria-hidden="true" className="w-4 h-4" />
        </a>
      </motion.div>

      {/* Phase pills */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8, duration: 0.8 }}
        className="relative z-10 mt-14 flex flex-wrap justify-center gap-2.5"
      >
        {phases.map((phase, i) => (
          <motion.span
            key={phase.label}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.9 + i * 0.08, type: "spring", stiffness: 400, damping: 20 }}
            className={`px-4 py-1.5 rounded-full text-xs font-mono font-semibold ${phase.color} ${phase.bg} border ${phase.border} cursor-default`}
          >
            {phase.label}
          </motion.span>
        ))}
      </motion.div>

      {/* Stats row */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.2, duration: 0.6 }}
        className="relative z-10 mt-16 grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-2xl w-full"
      >
        {stats.map((stat) => (
          <div
            key={stat.label}
            className="text-center p-3 rounded-xl bg-black/[0.02] dark:bg-white/[0.02] border border-black/[0.06] dark:border-white/[0.04]"
          >
            <p className="text-lg font-bold font-mono text-gradient-neural">
              {stat.value}
            </p>
            <p className="text-[10px] text-slate-400 uppercase tracking-wider mt-0.5">
              {stat.label}
            </p>
          </div>
        ))}
      </motion.div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.6 }}
        aria-hidden="true" className="absolute bottom-8 z-10 flex flex-col items-center gap-2 animate-smooth-bounce"
      >
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">
          Scroll
        </span>
        <ChevronDown aria-hidden="true" className="w-4 h-4 text-slate-400" />
      </motion.div>
    </section>
  );
}
