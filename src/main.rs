use notan::draw::*;
use notan::prelude::*;
use tch::Tensor;
mod world;
use crate::world::World;

// #[notan_main] // uncomment to test notan window
fn main() -> Result<(), String> {
    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };
    println!("Device used: {:?}", device);
    let mut map = World::new(
        3,
        3,
        2,
        1,
        &[0.5, 0.9],
        &[0.1, 0.3],
        device,
        tch::Kind::Float,
    );
    println!("Before update:");
    map.print();
    map.update();
    println!("After update:");
    map.print();
    todo!()
    // notan::init().draw(draw).add_config(DrawConfig).build()
}

fn draw(gfx: &mut Graphics) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);
    draw.triangle((400.0, 100.0), (100.0, 500.0), (700.0, 500.0));
    gfx.render(&draw);
}
