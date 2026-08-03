"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronUp, ChevronDown } from "lucide-react";

const SECTIONS = [
  { id: "hero", label: "Hero" },
  { id: "demo", label: "Demo" },
  { id: "architecture", label: "Architecture" },
  { id: "quickstart", label: "Quick Start" },
  { id: "playground", label: "Playground" },
  { id: "resilience", label: "Resilience Lab" },
  { id: "training", label: "Training Lab" },
  { id: "pipeline", label: "Pipeline" },
  { id: "upload", label: "Upload" },
  { id: "viewer", label: "Viewer" },
  { id: "status", label: "Status" },
];

export default function ScrollNavigator() {
  const [activeSection, setActiveSection] = useState("hero");
  const [isVisible, setIsVisible] = useState(false);
  const [isUserScrolling, setIsUserScrolling] = useState(true);
  const hideTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const scrollLockTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isProgrammaticScrolling = useRef(false);

  useEffect(() => {
    const handleScroll = () => {
      const scrollPosition = window.scrollY;
      const viewportHeight = window.innerHeight;

      // 1. Show/Hide control panel based on fold position
      setIsVisible(scrollPosition > viewportHeight * 0.2);

      // Fade-in effect logic
      setIsUserScrolling(true);
      if (hideTimeoutRef.current) clearTimeout(hideTimeoutRef.current);
      hideTimeoutRef.current = setTimeout(() => {
        setIsUserScrolling(false);
      }, 2000);

      // 2. Guard condition: Don't fight active animations
      if (isProgrammaticScrolling.current) return;

      // 3. Mathematical Section Tracking
      // Looks at what section occupies the upper-middle focal point of the screen
      const triggerPoint = scrollPosition + viewportHeight / 3;

      for (let i = 0; i < SECTIONS.length; i++) {
        const el = document.getElementById(SECTIONS[i].id);
        if (!el) continue;

        const top = el.offsetTop;
        const bottom = top + el.offsetHeight;

        if (triggerPoint >= top && triggerPoint <= bottom) {
          setActiveSection(SECTIONS[i].id);
          break;
        }
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    // Initial call to set active section on load
    handleScroll();

    return () => {
      window.removeEventListener("scroll", handleScroll);
      if (hideTimeoutRef.current) clearTimeout(hideTimeoutRef.current);
      if (scrollLockTimeout.current) clearTimeout(scrollLockTimeout.current);
    };
  }, []);

  const currentIndex = SECTIONS.findIndex((s) => s.id === activeSection);

  const scrollToSection = (id: string, index: number) => {
    const el = document.getElementById(id);
    if (!el) return;

    // Immediately set active state to prevent hanging arrow visual locks
    setActiveSection(id);

    // Clear any existing scroll lock timeout to prevent race condition
    if (scrollLockTimeout.current) clearTimeout(scrollLockTimeout.current);
    isProgrammaticScrolling.current = true;

    const navbarOffset = 80;
    const offsetPosition = el.offsetTop - navbarOffset;

    window.scrollTo({
      top: offsetPosition,
      behavior: "smooth",
    });

    // Release scroll lock once smooth animation completes
    scrollLockTimeout.current = setTimeout(() => {
      isProgrammaticScrolling.current = false;
    }, 800);
  };

  const handleNext = () => {
    if (currentIndex < SECTIONS.length - 1) {
      const nextIndex = currentIndex + 1;
      scrollToSection(SECTIONS[nextIndex].id, nextIndex);
    }
  };

  const handlePrev = () => {
    if (currentIndex > 0) {
      const prevIndex = currentIndex - 1;
      scrollToSection(SECTIONS[prevIndex].id, prevIndex);
    }
  };

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ 
            opacity: isUserScrolling ? 1 : 0.35, 
            y: 0,
            scale: isUserScrolling ? 1 : 0.97
          }}
          exit={{ opacity: 0, y: 20 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          className="fixed bottom-6 right-6 z-50 max-md:right-0 max-md:left-0 max-md:bottom-4 max-md:px-4 flex justify-center items-center pointer-events-none"
        >
          <div className="pointer-events-auto flex md:flex-col items-center bg-zinc-950/70 backdrop-blur-xl border border-zinc-800/80 rounded-full md:p-2 shadow-[0_0_40px_rgba(0,0,0,0.7)] text-white max-md:px-4 max-md:py-2 max-md:gap-4 max-md:w-full max-md:max-w-md max-md:justify-between">
            
            {/* Arrow Up */}
            <button
              onClick={handlePrev}
              disabled={currentIndex === 0}
              className="p-2 rounded-full hover:bg-zinc-800/80 text-zinc-400 hover:text-white transition-all disabled:opacity-10 disabled:pointer-events-none active:scale-90"
              title="Previous Section"
              aria-label="Previous Section"
            >
              <ChevronUp size={18} className="max-md:-rotate-90" />
            </button>

            {/* Dots Menu (Desktop Layout) */}
            <div className="hidden md:flex flex-col gap-3.5 my-4 px-1.5">
              {SECTIONS.map((sec, idx) => {
                const isActive = sec.id === activeSection;
                return (
                  <button
                    key={sec.id}
                    onClick={() => scrollToSection(sec.id, idx)}
                    className="relative group flex items-center justify-center cursor-pointer"
                    aria-label={`Scroll to ${sec.label}`}
                  >
                    <span className="absolute right-8 bg-zinc-900 border border-zinc-800 text-zinc-200 text-xs px-2.5 py-1 rounded-md opacity-0 scale-95 group-hover:opacity-100 group-hover:scale-100 pointer-events-none transition-all duration-150 whitespace-nowrap shadow-xl">
                      {sec.label}
                    </span>
                    
                    <div className="w-4 h-4 flex items-center justify-center">
                      <motion.div
                        animate={{
                          scale: isActive ? 1.4 : 1,
                          backgroundColor: isActive ? "#ffffff" : "rgba(113, 113, 122, 0.4)",
                          boxShadow: isActive ? "0 0 10px rgba(255,255,255,0.6)" : "none"
                        }}
                        className="w-1.5 h-1.5 rounded-full transition-colors duration-100"
                      />
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Title Tracker Badge (Mobile Layout) */}
            <div className="md:hidden flex items-center gap-2">
              <span className="text-xs tracking-wider text-zinc-500 uppercase font-mono">Section</span>
              <span className="text-xs font-semibold bg-zinc-800/50 text-zinc-200 px-3 py-1 rounded-md border border-zinc-700/30 min-w-[110px] text-center">
                {SECTIONS[currentIndex]?.label || "Loading..."}
              </span>
            </div>

            {/* Arrow Down */}
            <button
              onClick={handleNext}
              disabled={currentIndex === SECTIONS.length - 1}
              className="p-2 rounded-full hover:bg-zinc-800/80 text-zinc-400 hover:text-white transition-all disabled:opacity-10 disabled:pointer-events-none active:scale-90"
              title="Next Section"
              aria-label="Next Section"
            >
              <ChevronDown size={18} className="max-md:-rotate-90" />
            </button>

          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}