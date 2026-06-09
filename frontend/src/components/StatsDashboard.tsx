import { useSimStore } from "../state/store";

interface StatCardProps {
  label: string;
  value: string | number;
  color?: string;
  sub?: string;
}

function StatCard({ label, value, color = "text-accent-blue", sub }: StatCardProps) {
  return (
    <div className="bg-bg-secondary rounded-lg border border-border-subtle p-3 flex flex-col gap-1 min-w-0">
      <span className="text-xs text-gray-500 uppercase tracking-wider truncate">
        {label}
      </span>
      <span className={`text-2xl font-semibold ${color} leading-tight truncate`}>
        {value}
      </span>
      {sub && <span className="text-xs text-gray-600">{sub}</span>}
    </div>
  );
}

function ModeTag({ mode }: { mode: string }) {
  const colors: Record<string, string> = {
    WATCH: "bg-accent-blue/20 text-accent-blue border-accent-blue/40",
    TURBO: "bg-accent-orange/20 text-accent-orange border-accent-orange/40",
    REPLAY: "bg-accent-purple/20 text-accent-purple border-accent-purple/40",
    PAUSED: "bg-gray-700/20 text-gray-400 border-gray-600/40",
  };
  const cls = colors[mode] ?? colors["PAUSED"];
  return (
    <span
      className={`text-xs px-2 py-0.5 rounded border font-semibold uppercase tracking-widest ${cls}`}
    >
      {mode}
    </span>
  );
}

export function StatsDashboard() {
  const { latestStats, connected } = useSimStore();

  if (!latestStats) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <span className={`w-2 h-2 rounded-full ${connected ? "bg-accent-green animate-pulse" : "bg-gray-600"}`} />
        {connected ? "Waiting for data..." : "Disconnected — retrying..."}
      </div>
    );
  }

  const s = latestStats;
  const alivePercent =
    s.population_size > 0
      ? Math.round((s.birds_alive / s.population_size) * 100)
      : 0;

  return (
    <div className="flex flex-col gap-3">
      {/* Header row */}
      <div className="flex items-center gap-3">
        <span
          className={`w-2 h-2 rounded-full ${
            connected ? "bg-accent-green animate-pulse" : "bg-gray-600"
          }`}
        />
        <span className="text-sm text-gray-400">Live</span>
        <ModeTag mode={s.mode} />
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-3 gap-2">
        <StatCard
          label="Generation"
          value={s.generation}
          color="text-accent-blue"
        />
        <StatCard
          label="Birds Alive"
          value={`${s.birds_alive} / ${s.population_size}`}
          color="text-accent-green"
          sub={`${alivePercent}%`}
        />
        <StatCard
          label="Ticks"
          value={s.ticks_this_gen}
          color="text-gray-300"
        />
        <StatCard
          label="Best Fitness"
          value={s.best_fitness.toFixed(0)}
          color="text-accent-yellow"
        />
        <StatCard
          label="Avg Fitness"
          value={s.avg_fitness.toFixed(0)}
          color="text-accent-purple"
        />
        <StatCard
          label="All-Time Best"
          value={s.best_fitness_ever.toFixed(0)}
          color="text-accent-orange"
          sub="↑ record"
        />
      </div>
    </div>
  );
}
