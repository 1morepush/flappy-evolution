import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { useSimStore } from "../state/store";

const CHART_STYLE = {
  backgroundColor: "#161b22",
  border: "1px solid #30363d",
  borderRadius: 8,
  padding: "8px 12px",
  fontSize: 11,
  fontFamily: "JetBrains Mono, monospace",
  color: "#e6edf3",
};

export function AnalyticsCharts() {
  const { generationHistory } = useSimStore();

  if (generationHistory.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-gray-600 text-sm">
        Analytics will appear after the first generation completes.
      </div>
    );
  }

  const data = generationHistory.map((r) => ({
    gen: r.generation,
    best: Math.round(r.best_fitness),
    avg: Math.round(r.avg_fitness),
    survival: r.max_survival_ticks,
    improvement: +(r.improvement_rate * 100).toFixed(1),
  }));

  return (
    <div className="flex flex-col gap-4">
      {/* Fitness over generations */}
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
          Fitness Over Generations
        </p>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis
              dataKey="gen"
              stroke="#8b949e"
              tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }}
              label={{ value: "Gen", position: "insideBottomRight", offset: -4, fill: "#8b949e", fontSize: 9 }}
            />
            <YAxis
              stroke="#8b949e"
              tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }}
              width={40}
            />
            <Tooltip contentStyle={CHART_STYLE} />
            <Legend
              wrapperStyle={{ fontSize: 10, fontFamily: "JetBrains Mono" }}
            />
            <Line
              type="monotone"
              dataKey="best"
              stroke="#e3b341"
              strokeWidth={2}
              dot={false}
              name="Best"
            />
            <Line
              type="monotone"
              dataKey="avg"
              stroke="#bc8cff"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="4 2"
              name="Average"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Survival time */}
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
          Max Survival Ticks
        </p>
        <ResponsiveContainer width="100%" height={120}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis
              dataKey="gen"
              stroke="#8b949e"
              tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }}
            />
            <YAxis
              stroke="#8b949e"
              tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }}
              width={40}
            />
            <Tooltip contentStyle={CHART_STYLE} />
            <Line
              type="monotone"
              dataKey="survival"
              stroke="#39d353"
              strokeWidth={2}
              dot={false}
              name="Survival"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Improvement rate */}
      <div>
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
          Improvement Rate (%)
        </p>
        <ResponsiveContainer width="100%" height={100}>
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" />
            <XAxis
              dataKey="gen"
              stroke="#8b949e"
              tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }}
            />
            <YAxis
              stroke="#8b949e"
              tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }}
              width={40}
            />
            <Tooltip contentStyle={CHART_STYLE} formatter={(v) => [`${v}%`, "Improvement"]} />
            <Line
              type="monotone"
              dataKey="improvement"
              stroke="#58a6ff"
              strokeWidth={1.5}
              dot={false}
              name="Improvement %"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
