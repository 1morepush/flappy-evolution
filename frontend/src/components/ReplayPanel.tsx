import { useEffect, useState } from "react";
import type { ControlMessage, ReplaySummary } from "../lib/types";

const API = "http://localhost:8000";

interface Props {
  send: (msg: ControlMessage) => void;
}

export function ReplayPanel({ send }: Props) {
  const [replays, setReplays] = useState<ReplaySummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeGen, setActiveGen] = useState<number | null>(null);

  const fetchReplays = () => {
    setLoading(true);
    fetch(`${API}/api/replays`)
      .then((r) => r.json())
      .then((data) => setReplays(data.replays ?? []))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchReplays();
    const id = setInterval(fetchReplays, 5000);
    return () => clearInterval(id);
  }, []);

  const loadReplay = (gen: number) => {
    setActiveGen(gen);
    send({ type: "load_replay", generation: gen });
  };

  return (
    <div className="bg-bg-secondary rounded-lg border border-border-subtle p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <p className="text-xs text-gray-500 uppercase tracking-wider">Replays</p>
        <button
          onClick={fetchReplays}
          className="text-xs text-accent-blue hover:text-accent-blue/80"
        >
          ↻ Refresh
        </button>
      </div>

      {loading && <p className="text-xs text-gray-600">Loading...</p>}

      {!loading && replays.length === 0 && (
        <p className="text-xs text-gray-600">
          No replays saved yet. Replays are saved at the end of each generation.
        </p>
      )}

      <div className="flex flex-col gap-1 max-h-48 overflow-y-auto">
        {replays
          .slice()
          .reverse()
          .map((r) => (
            <button
              key={r.generation}
              onClick={() => loadReplay(r.generation)}
              className={`flex justify-between items-center px-2 py-1.5 rounded text-xs border transition-colors ${
                activeGen === r.generation
                  ? "bg-accent-purple/20 border-accent-purple/50 text-accent-purple"
                  : "bg-bg-tertiary border-border-subtle text-gray-300 hover:border-accent-purple/40"
              }`}
            >
              <span>Gen {r.generation}</span>
              <span className="text-accent-yellow font-semibold">
                {r.best_fitness.toFixed(0)} pts
              </span>
              <span className="text-gray-600">▶ Replay</span>
            </button>
          ))}
      </div>

      {activeGen !== null && (
        <p className="text-xs text-accent-purple">
          Replaying Gen {activeGen} — switch to WATCH to observe.
        </p>
      )}
    </div>
  );
}
