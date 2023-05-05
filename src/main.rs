use notan::draw::*;
use notan::prelude::*;
use tch::Tensor;
mod creature;
mod utils;
mod world;
use crate::world::World;
use crate::world::WorldObject;
use tch::IndexOp;

// #[notan_main] // uncomment to test notan window
fn main() -> Result<(), String> {
    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };
    println!("Device used: {:?}", device);
    // Create a dummy world
    let mut world = World::new(
        3,
        3,
        2,
        1,
        &[0.5, 0.9],
        &[10., 10.],
        device,
        tch::Kind::Float,
        0,
    );
    // Create some dummy creatures
    let square = world::Square::new(2, &world);
    let mut square2 = world::Square::new(2, &world);
    square2.set_position(1., 1.);
    world.add_entity(&square);
    world.add_entity(&square2);
    world.print();
    for _ in 0..10 {
        world.update();
        println!("\nAfter update:");
        world.print();
    }
    // This code block is missing a return statement, so we will add a dummy return value
    Ok(())
    // notan::init().draw(draw).add_config(DrawConfig).build()
}

fn draw(gfx: &mut Graphics) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);
    draw.triangle((400.0, 100.0), (100.0, 500.0), (700.0, 500.0));
    gfx.render(&draw);
}
