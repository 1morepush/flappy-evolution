import { useState } from "react";
import type { ControlMessage } from "../lib/types";
import { useSimStore } from "../state/store";

interface Props {
  send: (msg: ControlMessage) => void;
}

function Btn({
  onClick,
  active,
  children,
  color = "blue",
  title,
}: {
  onClick: () => void;
  active?: boolean;
  children: React.ReactNode;
  color?: "blue" | "orange" | "red" | "green" | "gray";
  title?: string;
}) {
  const base = "px-3 py-1.5 rounded text-xs font-semibold border transition-colors";
  const colors: Record<string, string> = {
    blue: active
      ? "bg-accent-blue text-bg-primary border-accent-blue"
      : "bg-bg-tertiary text-accent-blue border-accent-blue/40 hover:border-accent-blue",
    orange: active
      ? "bg-accent-orange text-bg-primary border-accent-orange"
      : "bg-bg-tertiary text-accent-orange border-accent-orange/40 hover:border-accent-orange",
    red: "bg-bg-tertiary text-accent-orange border-accent-orange/40 hover:bg-accent-orange/10",
    green: "bg-bg-tertiary text-accent-green border-accent-green/40 hover:bg-accent-green/10",
    gray: "bg-bg-tertiary text-gray-400 border-border-subtle hover:text-gray-200",
  };
  return (
    <button className={`${base} ${colors[color]}`} onClick={onClick} title={title}>
      {children}
    </button>
  );
}

function Slider({
  label,
  min,
  max,
  step,
  value,
  onChange,
  format,
}: {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs text-gray-500">
        <span>{label}</span>
        <span className="text-gray-300">{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-accent-blue h-1"
      />
    </div>
  );
}

export function ControlBar({ send }: Props) {
  const { latestStats } = useSimStore();
  const mode = latestStats?.mode ?? "WATCH";
  const paused = mode === "PAUSED";

  const [turboSteps, setTurboSteps] = useState(30);
  const [mutRate, setMutRate] = useState(0.1);
  const [mutSigma, setMutSigma] = useState(0.3);

  const applyMutation = () => {
    send({ type: "set_mutation", rate: mutRate, sigma: mutSigma });
  };

  return (
    <div className="bg-bg-secondary rounded-lg border border-border-subtle p-3 flex flex-col gap-3">
      <p className="text-xs text-gray-500 uppercase tracking-wider">Controls</p>

      {/* Mode + pause */}
      <div className="flex flex-wrap gap-2">
        <Btn
          onClick={() => send({ type: "set_mode", mode: "WATCH" })}
          active={mode === "WATCH"}
          color="blue"
          title="60fps live view"
        >
          👁 WATCH
        </Btn>
        <Btn
          onClick={() => send({ type: "set_mode", mode: "TURBO" })}
          active={mode === "TURBO"}
          color="orange"
          title="Max speed training"
        >
          ⚡ TURBO
        </Btn>
        <Btn
          onClick={() => (paused ? send({ type: "resume" }) : send({ type: "pause" }))}
          color="gray"
        >
          {paused ? "▶ Resume" : "⏸ Pause"}
        </Btn>
        <Btn
          onClick={() => {
            if (window.confirm("Reset all training progress?")) {
              send({ type: "reset" });
            }
          }}
          color="red"
          title="Reset all training"
        >
          ↺ Reset
        </Btn>
      </div>

      {/* Turbo speed slider */}
      <Slider
        label="Turbo Steps / Loop"
        min={1}
        max={200}
        step={1}
        value={turboSteps}
        onChange={(v) => {
          setTurboSteps(v);
          send({ type: "set_speed", turbo_steps: Math.round(v) });
        }}
        format={(v) => `${Math.round(v)}×`}
      />

      {/* Mutation controls */}
      <div className="flex flex-col gap-2 border-t border-border-subtle pt-2">
        <p className="text-xs text-gray-500 uppercase tracking-wider">Mutation</p>
        <Slider
          label="Rate"
          min={0.01}
          max={1.0}
          step={0.01}
          value={mutRate}
          onChange={setMutRate}
          format={(v) => `${(v * 100).toFixed(0)}%`}
        />
        <Slider
          label="Sigma"
          min={0.01}
          max={2.0}
          step={0.01}
          value={mutSigma}
          onChange={setMutSigma}
          format={(v) => v.toFixed(2)}
        />
        <Btn onClick={applyMutation} color="green">
          Apply Mutation
        </Btn>
      </div>
    </div>
  );
}
