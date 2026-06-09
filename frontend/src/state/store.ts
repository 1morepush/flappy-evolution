import { create } from "zustand";
import type {
  StateFrame,
  StatsFrame,
  GenerationRecord,
  TopologyResponse,
} from "../lib/types";

interface SimStore {
  // Latest frames
  latestState: StateFrame | null;
  latestStats: StatsFrame | null;
  generationHistory: GenerationRecord[];
  topology: TopologyResponse | null;

  // Connection
  connected: boolean;

  // Actions
  setLatestState: (f: StateFrame) => void;
  setLatestStats: (f: StatsFrame) => void;
  addGenerationRecord: (r: GenerationRecord) => void;
  setTopology: (t: TopologyResponse) => void;
  setConnected: (v: boolean) => void;
  resetHistory: () => void;
}

export const useSimStore = create<SimStore>((set) => ({
  latestState: null,
  latestStats: null,
  generationHistory: [],
  topology: null,
  connected: false,

  setLatestState: (f) => set({ latestState: f }),
  setLatestStats: (f) => set({ latestStats: f }),
  addGenerationRecord: (r) =>
    set((s) => ({
      generationHistory: [...s.generationHistory.slice(-500), r],
    })),
  setTopology: (t) => set({ topology: t }),
  setConnected: (v) => set({ connected: v }),
  resetHistory: () =>
    set({ generationHistory: [], latestState: null, latestStats: null }),
}));
