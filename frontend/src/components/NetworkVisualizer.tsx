import { useEffect, useRef } from "react";
import { useSimStore } from "../state/store";

const LAYER_LABELS = [
  ["Bird Y", "Velocity", "Pipe Dist", "Gap Top", "Gap Bot"],
  ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"],
  ["Nothing", "Flap"],
];

const LAYER_COLORS = ["#58a6ff", "#bc8cff", "#39d353"];

const NODE_R = 14;
const W = 340;
const H = 280;

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

export function NetworkVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { latestState } = useSimStore();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const acts = latestState?.focus_activations;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#161b22";
    ctx.fillRect(0, 0, W, H);

    const layers = [5, 8, 2];
    const totalLayers = layers.length;
    const layerXs = layers.map(
      (_, i) => (W / (totalLayers + 1)) * (i + 1)
    );

    // Compute node positions
    const nodePos: Array<Array<{ x: number; y: number }>> = layers.map(
      (n, li) =>
        Array.from({ length: n }, (_, ni) => ({
          x: layerXs[li],
          y: (H / (n + 1)) * (ni + 1),
        }))
    );

    // Draw edges with weight colouring
    for (let li = 0; li < totalLayers - 1; li++) {
      const srcNodes = nodePos[li];
      const dstNodes = nodePos[li + 1];
      srcNodes.forEach((src, si) => {
        dstNodes.forEach((dst, di) => {
          const act =
            acts && acts[li] ? sigmoid(acts[li][si]) : 0.5;
          const dstAct =
            acts && acts[li + 1] ? sigmoid(acts[li + 1][di]) : 0.5;
          const alpha = (act * dstAct * 0.8 + 0.1).toFixed(2);
          const positive = (acts?.[li]?.[si] ?? 0) >= 0;
          ctx.strokeStyle = positive
            ? `rgba(88,166,255,${alpha})`
            : `rgba(247,129,102,${alpha})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(src.x, src.y);
          ctx.lineTo(dst.x, dst.y);
          ctx.stroke();
        });
      });
    }

    // Draw nodes
    layers.forEach((n, li) => {
      for (let ni = 0; ni < n; ni++) {
        const { x, y } = nodePos[li][ni];
        const rawAct = acts?.[li]?.[ni] ?? 0;
        const intensity = sigmoid(rawAct);
        const color = LAYER_COLORS[li];

        // Glow
        const glow = Math.round(intensity * 20);
        ctx.shadowBlur = glow;
        ctx.shadowColor = color;

        // Node fill
        ctx.beginPath();
        ctx.arc(x, y, NODE_R, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${hexToRgb(color)},${(0.15 + intensity * 0.85).toFixed(2)})`;
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Activation text
        ctx.fillStyle = "#fff";
        ctx.font = "8px JetBrains Mono, monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(rawAct.toFixed(1), x, y);

        // Label
        const label = LAYER_LABELS[li]?.[ni] ?? "";
        ctx.fillStyle = "#8b949e";
        ctx.font = "7px JetBrains Mono, monospace";
        ctx.textAlign = li === 0 ? "right" : li === totalLayers - 1 ? "left" : "center";
        const labelX =
          li === 0 ? x - NODE_R - 4 : li === totalLayers - 1 ? x + NODE_R + 4 : x;
        const labelY = li === 1 ? y + NODE_R + 9 : y;
        ctx.fillText(label, labelX, labelY);
      }
    });

    // Layer titles
    const titles = ["Input", "Hidden", "Output"];
    titles.forEach((t, li) => {
      ctx.fillStyle = LAYER_COLORS[li];
      ctx.font = "9px JetBrains Mono, monospace";
      ctx.textAlign = "center";
      ctx.fillText(t, layerXs[li], 12);
    });
  }, [latestState]);

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-gray-500 uppercase tracking-wider">
        Neural Network — Focus Bird
      </span>
      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        className="w-full rounded-lg border border-border-subtle bg-bg-secondary"
        style={{ aspectRatio: `${W}/${H}` }}
      />
    </div>
  );
}

function hexToRgb(hex: string): string {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return "255,255,255";
  return `${parseInt(result[1], 16)},${parseInt(result[2], 16)},${parseInt(result[3], 16)}`;
}
