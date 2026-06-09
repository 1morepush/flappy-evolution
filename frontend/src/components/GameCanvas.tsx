import { useEffect, useRef, useCallback } from "react";
import { useSimStore } from "../state/store";
import type { StateFrame } from "../lib/types";

// Game world dimensions (must match backend GameConfig)
const W = 800;
const H = 512;
const GROUND_Y = 480;
const BIRD_X = 120;
const BIRD_R = 12;
const PIPE_WIDTH = 60;

interface Props {
  onBirdClick?: (id: number) => void;
}

export function GameCanvas({ onBirdClick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const frameRef = useRef<StateFrame | null>(null);
  const { latestState } = useSimStore();

  // Keep frame ref in sync with store (avoids stale closure in rAF)
  useEffect(() => {
    frameRef.current = latestState;
  }, [latestState]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const frame = frameRef.current;

    // Background
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, W, H);

    // Stars / scanlines effect
    ctx.fillStyle = "rgba(255,255,255,0.015)";
    for (let y = 0; y < H; y += 4) {
      ctx.fillRect(0, y, W, 1);
    }

    if (!frame) {
      // Waiting for connection
      ctx.fillStyle = "#58a6ff";
      ctx.font = "14px JetBrains Mono, monospace";
      ctx.textAlign = "center";
      ctx.fillText("Connecting to simulation...", W / 2, H / 2);
      animRef.current = requestAnimationFrame(draw);
      return;
    }

    // Draw pipes
    frame.pipes.forEach((pipe) => {
      const px = pipe.x;
      const left = px - PIPE_WIDTH / 2;

      // Pipe colour with glow
      ctx.shadowBlur = 8;
      ctx.shadowColor = "#39d35388";
      ctx.fillStyle = "#1a3a2a";

      // Top pipe (from y=0 down to gap_top)
      ctx.fillRect(left, 0, PIPE_WIDTH, pipe.gap_top);
      // Pipe cap
      ctx.fillStyle = "#2ea043";
      ctx.fillRect(left - 4, pipe.gap_top - 16, PIPE_WIDTH + 8, 16);

      // Bottom pipe (from gap_bottom down to ground)
      ctx.fillStyle = "#1a3a2a";
      ctx.fillRect(left, pipe.gap_bottom, PIPE_WIDTH, GROUND_Y - pipe.gap_bottom);
      // Pipe cap
      ctx.fillStyle = "#2ea043";
      ctx.fillRect(left - 4, pipe.gap_bottom, PIPE_WIDTH + 8, 16);

      ctx.shadowBlur = 0;
    });

    // Ground
    ctx.fillStyle = "#30363d";
    ctx.fillRect(0, GROUND_Y, W, H - GROUND_Y);
    ctx.fillStyle = "#444c56";
    ctx.fillRect(0, GROUND_Y, W, 2);

    // Birds
    const birds = frame.birds;
    const P = birds.ys.length;
    for (let i = 0; i < P; i++) {
      if (!birds.alive[i]) continue;

      const isFocus = i === frame.focus_id;

      if (isFocus) {
        // Focused bird — bright yellow with glow
        ctx.shadowBlur = 16;
        ctx.shadowColor = "#e3b341";
        ctx.fillStyle = "#e3b341";
        ctx.beginPath();
        ctx.arc(BIRD_X, birds.ys[i], BIRD_R, 0, Math.PI * 2);
        ctx.fill();

        // Eye
        ctx.shadowBlur = 0;
        ctx.fillStyle = "#0d1117";
        ctx.beginPath();
        ctx.arc(BIRD_X + 4, birds.ys[i] - 3, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(BIRD_X + 5, birds.ys[i] - 3, 1.5, 0, Math.PI * 2);
        ctx.fill();
      } else {
        // Other alive birds — semi-transparent blue
        ctx.shadowBlur = 0;
        ctx.fillStyle = "rgba(88, 166, 255, 0.35)";
        ctx.beginPath();
        ctx.arc(BIRD_X, birds.ys[i], BIRD_R - 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.shadowBlur = 0;

    // HUD overlay
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(8, 8, 160, 48);
    ctx.fillStyle = "#58a6ff";
    ctx.font = "11px JetBrains Mono, monospace";
    ctx.textAlign = "left";
    ctx.fillText(`Tick: ${frame.tick}`, 16, 26);
    ctx.fillText(
      `Alive: ${birds.alive.filter(Boolean).length} / ${P}`,
      16,
      44
    );

    animRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => {
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [draw]);

  // Click → focus nearest alive bird
  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!onBirdClick || !frameRef.current) return;
      const canvas = canvasRef.current!;
      const rect = canvas.getBoundingClientRect();
      const scaleX = W / rect.width;
      const scaleY = H / rect.height;
      const cx = (e.clientX - rect.left) * scaleX;
      const cy = (e.clientY - rect.top) * scaleY;

      const birds = frameRef.current.birds;
      let best = -1;
      let bestDist = Infinity;
      birds.ys.forEach((y, i) => {
        if (!birds.alive[i]) return;
        const d = Math.hypot(cx - BIRD_X, cy - y);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      });
      if (best >= 0) onBirdClick(best);
    },
    [onBirdClick]
  );

  return (
    <canvas
      ref={canvasRef}
      width={W}
      height={H}
      onClick={handleClick}
      className="w-full rounded-lg border border-border-subtle cursor-crosshair"
      style={{ aspectRatio: `${W}/${H}`, imageRendering: "pixelated" }}
    />
  );
}
