import { useEffect } from "react";
import { useSimSocket } from "./hooks/useSimSocket";
import { useSimStore } from "./state/store";
import { GameCanvas } from "./components/GameCanvas";
import { NetworkVisualizer } from "./components/NetworkVisualizer";
import { StatsDashboard } from "./components/StatsDashboard";
import { AnalyticsCharts } from "./components/AnalyticsCharts";
import { ControlBar } from "./components/ControlBar";
import { ReplayPanel } from "./components/ReplayPanel";

const API = "http://localhost:8000";

export default function App() {
  const { send } = useSimSocket();
  const setTopology = useSimStore((s) => s.setTopology);

  // Fetch static topology once
  useEffect(() => {
    fetch(`${API}/api/topology`)
      .then((r) => r.json())
      .then(setTopology)
      .catch(() => {});
  }, [setTopology]);

  const handleBirdClick = (id: number) => {
    send({ type: "focus_bird", bird_id: id });
  };

  return (
    <div className="min-h-screen bg-bg-primary text-gray-100 font-mono">
      {/* Header */}
      <header className="border-b border-border-subtle bg-bg-secondary px-4 py-3">
        <div className="max-w-screen-2xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🐦</span>
            <div>
              <h1 className="text-sm font-semibold text-white leading-tight">
                Flappy Evolution
              </h1>
              <p className="text-xs text-gray-500">
                Neural Network Neuroevolution Simulator
              </p>
            </div>
          </div>
          <div className="text-xs text-gray-600">
            5→8→2 · Genetic Algorithm · 100 birds/gen
          </div>
        </div>
      </header>

      {/* Main layout */}
      <main className="max-w-screen-2xl mx-auto p-3 flex flex-col gap-3">
        {/* Row 1: Canvas + Right sidebar */}
        <div className="flex gap-3 items-start">
          {/* Game Canvas */}
          <div className="flex-1 min-w-0">
            <GameCanvas onBirdClick={handleBirdClick} />
          </div>

          {/* Right sidebar */}
          <div className="w-72 flex-shrink-0 flex flex-col gap-3">
            <StatsDashboard />
            <ControlBar send={send} />
            <ReplayPanel send={send} />
          </div>
        </div>

        {/* Row 2: Analytics + NN Visualizer */}
        <div className="flex gap-3 items-start">
          {/* Analytics charts */}
          <div className="flex-1 min-w-0 bg-bg-secondary rounded-lg border border-border-subtle p-3">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">
              Training Analytics
            </p>
            <AnalyticsCharts />
          </div>

          {/* Neural network visualizer */}
          <div className="w-72 flex-shrink-0 bg-bg-secondary rounded-lg border border-border-subtle p-3">
            <NetworkVisualizer />
          </div>
        </div>
      </main>
    </div>
  );
}
