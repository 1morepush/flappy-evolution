# Flappy Evolution — Neural Network Neuroevolution Simulator

100 Flappy Bird agents, each controlled by a **5→8→2 feedforward neural network**, trained
generation-after-generation by a genetic algorithm — all running live in your browser.

---

## Quick Start

### 1. Backend (Python / FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend (React / Vite)

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

---

## Architecture

```
flappy-evolution/
├── backend/
│   ├── app/
│   │   ├── main.py          FastAPI app, REST + WebSocket
│   │   ├── config.py        All tuneable constants (Pydantic)
│   │   ├── schemas.py       Wire models (StateFrame, StatsFrame, etc.)
│   │   └── engine_runner.py Background simulation loop, mode control
│   ├── neural_network/      NumPy feedforward net (vectorised across 100 birds)
│   ├── evolution/           Genetic algorithm (elitism + tournament + crossover + mutation)
│   ├── game/                Physics engine + vectorised world
│   ├── analytics/           Per-generation statistics
│   ├── replays/             JSON replay store (best bird per generation)
│   └── tests/               26 pytest unit tests
└── frontend/
    └── src/
        ├── components/
        │   ├── GameCanvas.tsx       Canvas renderer (birds + pipes)
        │   ├── NetworkVisualizer.tsx  NN diagram with live activations
        │   ├── StatsDashboard.tsx   Live stats cards
        │   ├── AnalyticsCharts.tsx  Recharts analytics (Recharts)
        │   ├── ControlBar.tsx       Mode, speed, mutation controls
        │   └── ReplayPanel.tsx      Replay saved generations
        ├── hooks/useSimSocket.ts    WebSocket client with auto-reconnect
        └── state/store.ts          Zustand global store
```

> **Folder → requirement mapping:** `/neural_network`, `/evolution`, `/analytics`, `/replays`
> live under `backend/`. The "visualizations" requirement is served by `NetworkVisualizer.tsx`
> and `AnalyticsCharts.tsx` in the frontend.

---

## Neural Network

| Layer  | Neurons | Activation |
|--------|---------|------------|
| Input  | 5       | —          |
| Hidden | 8       | tanh       |
| Output | 2       | argmax     |

**Inputs (normalised 0–1):** Bird Y · Bird velocity · Pipe distance · Gap top · Gap bottom  
**Outputs:** Nothing (0) · Flap (1)  
**Genome:** 66 float32 genes (flat vector encoding all weights + biases)

All 100 birds are processed in a **single batched `np.einsum` call** — no per-bird Python
loops.

---

## Evolutionary Algorithm

1. **Elitism** — top 4 genomes copied unchanged
2. **Tournament selection** — 4-way tournament to pick parents
3. **Uniform crossover** — each gene drawn from parent A or B
4. **Gaussian mutation** — 10% of genes perturbed by N(0, 0.3)

**Fitness** = survival ticks + 10-point bonus per pipe passed.  
Each generation uses a fixed random seed so all birds face identical pipe layouts → fair
fitness comparison and deterministic replay.

---

## Modes

| Mode   | Description |
|--------|-------------|
| WATCH  | 60fps, every frame streamed to browser |
| TURBO  | Headless-fast, sampled snapshots (~150ms interval) |
| REPLAY | Plays back any saved generation's best bird deterministically |
| PAUSED | Loop suspended |

---

## REST API

| Endpoint                    | Description                            |
|-----------------------------|----------------------------------------|
| `GET /api/health`           | Server status + current generation    |
| `GET /api/topology`         | Network shape + label names            |
| `GET /api/analytics`        | Full generation history                |
| `GET /api/replays`          | List of saved generations              |
| `GET /api/replays/{gen}`    | Genome + seed for a specific generation|

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

26 unit tests cover: network forward pass, genome pack/unpack, mutation, crossover,
elitism, tournament selection, physics, collision detection, world simulation, and sensors.

---

## Configuration

All defaults live in `backend/app/config.py`:

```python
population_size = 100
elite_count = 4
tournament_size = 4
mutation_rate = 0.1
mutation_sigma = 0.3
pipe_gap = 140
turbo_steps_per_tick = 30
```

These are also controllable live from the UI (mutation rate, turbo speed).
