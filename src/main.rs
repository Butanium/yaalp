use notan::draw::*;
use notan::prelude::*;

mod constants;
mod plant;
mod world;
mod yaal;

use crate::world::World;

// #[notan_main] // uncomment to test notan window
fn main() -> Result<(), String> {
    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };
    println!("Device used: {:?}", device);
    println!("Starting YAALP - Yet Another Artificial Life Project");
    println!("====================================================\n");

    // Create world with 6 channels:
    // 0-2: RGB signature channels (for Yaal identification)
    // 3-5: Resource/food channels (with diffusion)
    let width = 100;
    let height = 100;
    let channels = 6;
    let max_fov = 3;

    // Decay factors: RGB channels don't decay, resource channels decay slowly
    let decays = [1.0, 1.0, 1.0, 0.99, 0.98, 0.97];
    // Max values: RGB capped at 1.0, resources can accumulate to 5.0
    let max_values = [1.0, 1.0, 1.0, 5.0, 5.0, 5.0];

    let mut world = World::new(
        width,
        height,
        channels,
        max_fov,
        &decays,
        &max_values,
        device,
        tch::Kind::Float,
    );

    println!(
        "Created world: {}x{} with {} channels",
        width, height, channels
    );

    // Create some Yaals and Plants
    let num_yaals = 10;
    let num_plants = 20;
    world.create_yaals_and_plants(num_yaals, num_plants);

    println!("Spawned {} Yaals and {} Plants\n", num_yaals, num_plants);

    // Run simulation
    let timesteps = 100;
    println!("Running simulation for {} timesteps...\n", timesteps);

    for t in 0..timesteps {
        world.step();

        // Print progress every 10 steps
        if t % 10 == 0 {
            println!("Step {}/{}", t, timesteps);
            println!(
                "  Alive Yaals: {} | Plants: {}",
                world.yaals.len(),
                world.plants.len()
            );

            // Show first Yaal's position if any exist
            if let Some(yaal) = world.yaals.first() {
                println!(
                    "  First Yaal at ({:.1}, {:.1})",
                    yaal.position.x(),
                    yaal.position.y()
                );
            }
        }
    }

    println!("\n✓ Simulation complete!");
    println!("Final statistics:");
    println!("  Total Yaals: {}", world.yaals.len());
    println!("  Total Plants: {}", world.plants.len());

    Ok(())
    // notan::init().draw(draw).add_config(DrawConfig).build()
}

fn draw(gfx: &mut Graphics) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);
    draw.triangle((400.0, 100.0), (100.0, 500.0), (700.0, 500.0));
    gfx.render(&draw);
}
