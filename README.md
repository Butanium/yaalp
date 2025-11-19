# Yaalp
Yet Another Artificial Life Project

An artificial life simulation where creatures (Yaals) evolve and interact in a tensor-based environment.

## Features

- **Yaals**: Autonomous creatures with:
  - Neural network brains for decision-making
  - Genetic genomes (speed, field of view, size, visual signature)
  - Movement based on environmental perception

- **Plants**: Food sources that populate the environment

- **Tensor-based World**: Multi-channel map system
  - Channels 0-2: RGB signature identification
  - Channels 3-5: Resources with decay over time
  - Support for PyTorch tensors (CPU/CUDA)

- **Simulation Features**:
  - Decay system for resources
  - Bounded world with collision handling
  - Configurable world parameters

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

## Synced with yaalpp

This Rust implementation is synced with the [yaalpp](https://github.com/Butanium/yaalpp) C++ version, sharing core features:
- Entity system (Yaals and Plants)
- Multi-channel tensor-based world
- Decay and resource systems
- Neural network brains for creatures
