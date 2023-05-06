use notan::draw::*;
use notan::prelude::*;
mod constants;
mod creature;
mod utils;
mod world;
use crate::creature::RandomInit;
use crate::world::World;
use tch::IndexOp;

// #[notan_main] // uncomment to test notan window
fn main() -> Result<(), String> {
    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };
    let _guard = tch::no_grad_guard(); // disable gradient calculation
    println!("Device used: {:?}", device);
    let seed = rand::random();
    println!("Seed used: {:?}", seed);
    // Create a dummy world
    let mut world = World::new(
        3,
        3,
        2,
        2,
        &[0.5, 0.9],
        &[10., 10.],
        device,
        tch::Kind::Float,
        seed,
    );
    // Create some dummy creatures
    let square = world::Square::new(2, &world);
    let yaal = creature::Yaal::new_random(&mut world);
    world.add_entity(&yaal);
    println!("Our Yaal: {:#?}", yaal);
    world.add_entity(&square);
    world.print();
    for i in 0..3 {
        world.update();
        println!("\nAfter update{:?}:", i);
        world.print();
    }
    // dummy return value
    Ok(())
    // notan::init().draw(draw).add_config(DrawConfig).build()
}
#[test]
fn smoke_test() {
    for _ in 0..100 {
        main().unwrap();
    }
}
// notan example
// fn draw(gfx: &mut Graphics) {
//     let mut draw = gfx.create_draw();
//     draw.clear(Color::BLACK);
//     draw.triangle((400.0, 100.0), (100.0, 500.0), (700.0, 500.0));
//     gfx.render(&draw);
// }
