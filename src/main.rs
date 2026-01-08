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
    let timesteps = 200;
    println!("Running simulation for {} timesteps...\n", timesteps);

    let mut total_deaths = 0;
    let mut total_births = 0;
    let mut total_food_consumed = 0;

    for t in 0..timesteps {
        let yaals_before = world.yaals.len();
        let plants_before = world.plants.len();

        world.step();

        let yaals_after = world.yaals.len();
        let plants_after = world.plants.len();

        // Calculate births and deaths
        let net_change = (yaals_after as i64) - (yaals_before as i64);
        let births_this_step = if net_change > 0 {
            net_change as usize
        } else {
            0
        };
        let deaths_this_step = if net_change < 0 {
            (-net_change) as usize
        } else {
            0
        };

        let food_consumed_this_step = plants_before.saturating_sub(plants_after);

        total_births += births_this_step;
        total_deaths += deaths_this_step;
        total_food_consumed += food_consumed_this_step;

        // Print progress every 20 steps
        if t % 20 == 0 || yaals_after == 0 {
            println!("\n━━━ Step {}/{} ━━━", t, timesteps);
            println!(
                "  🦠 Alive Yaals: {} | 🌱 Plants: {}",
                yaals_after, plants_after
            );

            if yaals_after > 0 {
                // Calculate average energy and health
                let total_energy: f32 = world.yaals.iter().map(|y| y.energy).sum();
                let total_health: f32 = world.yaals.iter().map(|y| y.health).sum();
                let avg_energy = total_energy / yaals_after as f32;
                let avg_health = total_health / yaals_after as f32;
                let avg_age: i64 =
                    world.yaals.iter().map(|y| y.age).sum::<i64>() / yaals_after as i64;

                println!(
                    "  📊 Avg Energy: {:.1} | Avg Health: {:.1} | Avg Age: {}",
                    avg_energy, avg_health, avg_age
                );

                // Show healthiest Yaal
                if let Some(healthiest) = world
                    .yaals
                    .iter()
                    .max_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
                {
                    println!(
                        "  🏆 Healthiest Yaal: Energy={:.1}, Age={}, Pos=({:.1}, {:.1})",
                        healthiest.energy,
                        healthiest.age,
                        healthiest.position.x(),
                        healthiest.position.y()
                    );
                }

                if deaths_this_step > 0 {
                    println!("  ⚠️  {} Yaals died this period", deaths_this_step);
                }
                if births_this_step > 0 {
                    println!("  🎉 {} new Yaals born!", births_this_step);
                }
            } else {
                println!("  ☠️  All Yaals have died!");
                break;
            }
        }
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("✓ Simulation complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Final statistics:");
    println!(
        "  Survivors: {}/{} Yaals (started with {})",
        world.yaals.len(),
        num_yaals + total_births,
        num_yaals
    );
    println!("  Total births: {} 🎉", total_births);
    println!("  Total deaths: {} ⚰️", total_deaths);
    println!("  Plants remaining: {} 🌱", world.plants.len());
    println!("  Total food consumed: {} 🍽️", total_food_consumed);

    if world.yaals.is_empty() {
        println!("\n  Population went extinct! 💀");
    } else {
        let oldest = world.yaals.iter().max_by_key(|y| y.age).unwrap();
        println!("\n  Oldest survivor: Age {}", oldest.age);
        let avg_energy: f32 =
            world.yaals.iter().map(|y| y.energy).sum::<f32>() / world.yaals.len() as f32;
        println!("  Average energy: {:.1}", avg_energy);
    }

    Ok(())
    // notan::init().draw(draw).add_config(DrawConfig).build()
}

fn draw(gfx: &mut Graphics) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);
    draw.triangle((400.0, 100.0), (100.0, 500.0), (700.0, 500.0));
    gfx.render(&draw);
}
