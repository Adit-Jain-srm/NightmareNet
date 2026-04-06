"use client";

import { useRef, useEffect, useCallback } from "react";

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  color: string;
  alpha: number;
  layer: number; // 0 = back, 1 = mid, 2 = front
}

const COLORS = {
  dream: "129,140,248",
  nightmare: "248,113,113",
  neural: "34,211,238",
  success: "52,211,153",
};

const COLOR_KEYS = Object.keys(COLORS) as (keyof typeof COLORS)[];

function createNode(w: number, h: number): Node {
  const layer = Math.random() < 0.3 ? 0 : Math.random() < 0.6 ? 1 : 2;
  const colorKey = COLOR_KEYS[Math.floor(Math.random() * COLOR_KEYS.length)];
  const speed = (0.15 + Math.random() * 0.25) * (layer === 0 ? 0.5 : layer === 1 ? 0.75 : 1);
  return {
    x: Math.random() * w,
    y: Math.random() * h,
    vx: (Math.random() - 0.5) * speed,
    vy: (Math.random() - 0.5) * speed,
    radius: layer === 0 ? 1 : layer === 1 ? 1.5 : 2,
    color: COLORS[colorKey],
    alpha: layer === 0 ? 0.15 : layer === 1 ? 0.3 : 0.5,
    layer,
  };
}

export default function NeuralBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<Node[]>([]);
  const mouseRef = useRef({ x: -1000, y: -1000 });
  const animRef = useRef<number>(0);
  const reducedMotion = useRef(false);

  const init = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    canvas.width = w;
    canvas.height = h;
    const nodeCount = Math.min(Math.floor((w * h) / 12000), 120);
    nodesRef.current = Array.from({ length: nodeCount }, () => createNode(w, h));
  }, []);

  useEffect(() => {
    reducedMotion.current = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    init();

    const handleResize = () => init();
    const handleMouse = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    window.addEventListener("resize", handleResize);
    window.addEventListener("mousemove", handleMouse);

    if (reducedMotion.current) {
      // Static render for reduced motion
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          nodesRef.current.forEach((n) => {
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(${n.color},${n.alpha})`;
            ctx.fill();
          });
        }
      }
      return () => {
        window.removeEventListener("resize", handleResize);
        window.removeEventListener("mousemove", handleMouse);
      };
    }

    const CONNECTION_DIST = 140;
    const MOUSE_RADIUS = 200;
    const MOUSE_FORCE = 0.3;

    const animate = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const { width: w, height: h } = canvas;

      ctx.clearRect(0, 0, w, h);

      const nodes = nodesRef.current;
      const mouse = mouseRef.current;

      // Update positions
      for (const node of nodes) {
        // Mouse interaction
        const dx = node.x - mouse.x;
        const dy = node.y - mouse.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < MOUSE_RADIUS && dist > 0) {
          const force = ((MOUSE_RADIUS - dist) / MOUSE_RADIUS) * MOUSE_FORCE;
          node.vx += (dx / dist) * force * 0.05;
          node.vy += (dy / dist) * force * 0.05;
        }

        // Damping
        node.vx *= 0.995;
        node.vy *= 0.995;

        node.x += node.vx;
        node.y += node.vy;

        // Wrap
        if (node.x < -20) node.x = w + 20;
        if (node.x > w + 20) node.x = -20;
        if (node.y < -20) node.y = h + 20;
        if (node.y > h + 20) node.y = -20;
      }

      // Draw connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < CONNECTION_DIST) {
            const alpha = ((CONNECTION_DIST - dist) / CONNECTION_DIST) * 0.12 *
              Math.min(a.alpha, b.alpha);
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `rgba(${a.color},${alpha})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      for (const node of nodes) {
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${node.color},${node.alpha})`;
        ctx.fill();

        // Glow
        if (node.layer === 2) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius * 3, 0, Math.PI * 2);
          const grad = ctx.createRadialGradient(
            node.x, node.y, node.radius,
            node.x, node.y, node.radius * 3
          );
          grad.addColorStop(0, `rgba(${node.color},${node.alpha * 0.3})`);
          grad.addColorStop(1, `rgba(${node.color},0)`);
          ctx.fillStyle = grad;
          ctx.fill();
        }
      }

      animRef.current = requestAnimationFrame(animate);
    };

    animRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", handleMouse);
      cancelAnimationFrame(animRef.current);
    };
  }, [init]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-0 pointer-events-none"
      aria-hidden="true"
      style={{ opacity: 0.7 }}
    />
  );
}
