# Yaalp
Yet Another Artificial Life Project

An artificial life simulation where creatures (Yaals) evolve and interact in a tensor-based environment.

## Features

- **Yaals**: Autonomous creatures with:
  - Neural network brains (MLP) for decision-making
  - Genetic genomes (speed, field of view, size, visual signature)
  - Movement based on environmental perception
  - **Energy and health system**: Creatures need to eat to survive
  - **Reproduction with mutation**: Asexual reproduction with genetic variation
  - **Natural selection**: Better adapted creatures are more likely to survive and reproduce

- **Plants**: Food sources that populate the environment
  - Consumed by Yaals for energy
  - Randomly respawn to maintain ecosystem balance

- **Ecosystem Dynamics**:
  - **Food chain**: Yaals hunt and consume plants
  - **Death**: Yaals die when energy or health depletes
  - **Birth**: Successful Yaals reproduce when well-fed
  - **Evolution**: Neural network weights mutate across generations
  - **Population dynamics**: Emergent behavior from survival pressures

- **Tensor-based World**: Multi-channel map system
  - Channels 0-2: RGB signature identification
  - Channels 3-5: Resources with decay over time
  - Support for PyTorch tensors (CPU/CUDA)

- **Simulation Features**:
  - Decay system for resources
  - Collision detection for food consumption
  - Bounded world with position clamping
  - Configurable world parameters
  - Real-time statistics tracking (births, deaths, energy, health, age)

## Requirements

In order to run this project, you need to have [Rust](https://www.rust-lang.org/tools/install) and the `LibTorch` library installed. Follow the instructions on the [tch-rust repo](https://github.com/LaurentMazare/tch-rs) to install `LibTorch`. For Linux installation, make sure to choose the `cxx11 ABI` and not the `Pre-cxx11 ABI` on the PyTorch website.

## Usage

To run the project, clone the repository and use `cargo run`.

```bash
git clone https://github.com/Butanium/yaalp
cd yaalp
cargo run
```

## Architecture

The project is organized into several modules:

- `world.rs`: Core world simulation with tensor-based map
- `yaal.rs`: Yaal creatures with genome and neural network brain
- `plant.rs`: Plant entities
- `constants.rs`: Simulation constants and parameters
- `main.rs`: Entry point and simulation loop

## Example Output

```
Starting YAALP - Yet Another Artificial Life Project
====================================================

Created world: 100x100 with 6 channels
Spawned 10 Yaals and 20 Plants

Running simulation for 200 timesteps...

━━━ Step 0/200 ━━━
  🦠 Alive Yaals: 10 | 🌱 Plants: 20
  📊 Avg Energy: 720.0 | Avg Health: 900.0 | Avg Age: 0
  🏆 Healthiest Yaal: Energy=720.0, Age=0, Pos=(45.2, 67.8)

━━━ Step 20/200 ━━━
  🦠 Alive Yaals: 12 | 🌱 Plants: 18
  📊 Avg Energy: 650.3 | Avg Health: 890.5 | Avg Age: 15
  🏆 Healthiest Yaal: Energy=780.2, Age=20, Pos=(32.1, 54.3)
  🎉 2 new Yaals born!

...

✓ Simulation complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Final statistics:
  Survivors: 15/28 Yaals (started with 10)
  Total births: 18 🎉
  Total deaths: 13 ⚰️
  Plants remaining: 22 🌱
  Total food consumed: 47 🍽️

  Oldest survivor: Age 198
  Average energy: 682.4
```

## Synced with yaalpp

This Rust implementation is synced with the [yaalpp](https://github.com/Butanium/yaalpp) C++ version, sharing core features:
- Entity system (Yaals and Plants)
- Multi-channel tensor-based world
- Decay and resource systems
- Neural network brains for creatures

## Beyond yaalpp

This Rust version includes additional features beyond the C++ implementation:
- Complete energy and health system
- Food consumption with collision detection
- Reproduction with genetic mutation
- Population dynamics and natural selection
- Comprehensive statistics tracking
