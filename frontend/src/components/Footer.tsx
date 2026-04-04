"use client";

import { Brain, GitBranch, Heart } from "lucide-react";

function LinkedInIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
    </svg>
  );
}

export default function Footer() {
  return (
    <footer className="relative border-t border-white/[0.04] py-12 px-6">
      {/* Gradient divider */}
      <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-neural/20 to-transparent" />

      <div className="max-w-5xl mx-auto">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-neural/5 border border-neural/10 flex items-center justify-center">
              <Brain className="w-4 h-4 text-neural" />
            </div>
            <div>
              <span className="font-mono font-bold text-sm">
                <span className="text-gradient-neural">Nightmare</span>
                <span className="text-text-dim">Net</span>
              </span>
              <p className="text-[10px] text-muted">Autonomous AI Self-Improvement</p>
            </div>
          </div>

          {/* Links */}
          <div className="flex items-center gap-6 text-xs text-muted">
            <a
              href="https://github.com/Adit-Jain-srm/NightmareNet"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-neural transition-colors flex items-center gap-1.5 cursor-pointer"
            >
              <GitBranch className="w-3.5 h-3.5" />
              GitHub
            </a>
            <span className="text-white/[0.06]">|</span>
            <a
              href="https://www.linkedin.com/in/-adit-jain"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-dream transition-colors flex items-center gap-1.5 cursor-pointer"
            >
              <LinkedInIcon className="w-3.5 h-3.5" />
              LinkedIn
            </a>
            <span className="text-white/[0.06]">|</span>
            <span className="flex items-center gap-1">
              Built with <Heart className="w-3 h-3 text-nightmare/50" /> by Adit Jain
            </span>
          </div>

          {/* Tech stack */}
          <div className="flex items-center gap-3">
            {["Next.js", "FastAPI", "PyTorch", "Framer"].map((tech) => (
              <span
                key={tech}
                className="text-[9px] font-mono text-muted/50 px-2 py-1 rounded-md bg-white/[0.02] border border-white/[0.03]"
              >
                {tech}
              </span>
            ))}
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-white/[0.03] text-center">
          <p className="text-[10px] text-muted/40 font-mono">
            NightmareNet v0.2.0 • MIT License • Sleep-Inspired Training Paradigm
          </p>
        </div>
      </div>
    </footer>
  );
}
