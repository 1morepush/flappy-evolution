// TypeScript mirrors of backend wire schemas (schemas.py)

export interface BirdsArray {
  ys: number[];
  vys: number[];
  alive: boolean[];
}

export interface PipeState {
  x: number;
  gap_top: number;
  gap_bottom: number;
}

export interface StateFrame {
  type: "state";
  tick: number;
  birds: BirdsArray;
  pipes: PipeState[];
  focus_activations: number[][];  // [[inputs], [hidden], [outputs]]
  focus_id: number;
}

export interface StatsFrame {
  type: "stats";
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  population_size: number;
  birds_alive: number;
  best_fitness_ever: number;
  mode: "WATCH" | "TURBO" | "REPLAY" | "PAUSED";
  ticks_this_gen: number;
}

export interface GenerationRecord {
  type: "generation";
  generation: number;
  best_fitness: number;
  avg_fitness: number;
  max_survival_ticks: number;
  improvement_rate: number;
  pipe_seed: number;
}

export type SimMessage = StateFrame | StatsFrame | GenerationRecord;

// REST types
export interface TopologyResponse {
  layers: number[];
  input_labels: string[];
  output_labels: string[];
}

export interface ReplaySummary {
  generation: number;
  best_fitness: number;
  pipe_seed: number;
}

export interface ReplayDetail {
  generation: number;
  best_fitness: number;
  pipe_seed: number;
  genome: number[];
  topology: number[];
}

// Control messages sent to server
export type ControlMessage =
  | { type: "set_mode"; mode: "WATCH" | "TURBO" }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "reset" }
  | { type: "set_speed"; turbo_steps: number }
  | { type: "focus_bird"; bird_id: number }
  | { type: "set_mutation"; rate: number; sigma: number }
  | { type: "load_replay"; generation: number };
