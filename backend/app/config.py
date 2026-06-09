"""Central configuration — all tuneable constants live here."""
from pydantic import BaseModel


class GameConfig(BaseModel):
    width: int = 800
    height: int = 512
    gravity: float = 0.5
    flap_impulse: float = -8.0
    pipe_speed: float = 3.0
    pipe_gap: int = 140          # vertical gap between top and bottom pipe
    pipe_spacing: int = 220      # horizontal distance between pipe pairs
    pipe_width: int = 60
    bird_x: int = 120            # fixed horizontal position of all birds
    bird_radius: int = 12
    ground_y: int = 480          # y-coordinate of the ground
    max_ticks_per_gen: int = 6000  # safety limit ~100s at 60fps


class NetworkConfig(BaseModel):
    input_size: int = 5
    hidden_size: int = 8
    output_size: int = 2
    topology: list[int] = [5, 8, 2]


class EvolutionConfig(BaseModel):
    population_size: int = 100
    elite_count: int = 4
    tournament_size: int = 4
    mutation_rate: float = 0.1    # fraction of genes to perturb
    mutation_sigma: float = 0.3   # std dev of Gaussian noise
    pipe_reward: float = 10.0     # bonus ticks for passing a pipe centre


class AppConfig(BaseModel):
    game: GameConfig = GameConfig()
    network: NetworkConfig = NetworkConfig()
    evolution: EvolutionConfig = EvolutionConfig()
    watch_fps: float = 60.0
    turbo_steps_per_tick: int = 30   # sim steps per loop iteration in TURBO
    turbo_snapshot_ms: float = 150   # ms between StateFrame snapshots in TURBO
    stats_every_ticks: int = 5       # broadcast StatsFrame every N ticks


# Singleton used throughout the app — mutated via REST control endpoint
CONFIG = AppConfig()
