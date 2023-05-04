use notan::draw::*;
use notan::prelude::*;
use tch::Tensor;
mod world;
use crate::world::World;

// #[notan_main]
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
    let tensor1 = Tensor::of_slice(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
    ])
    .reshape(&[2, 2, 2]);
    let tensor2 = Tensor::of_slice(&[2.0, 3.0]);
    let result = &tensor1 * (&tensor2.unsqueeze(-1).unsqueeze(-1));
    tensor1.print();
    result.print();
    map.print();
    map.update();
    todo!()
    // notan::init().draw(draw).add_config(DrawConfig).build()
}

fn draw(gfx: &mut Graphics) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);
    draw.triangle((400.0, 100.0), (100.0, 500.0), (700.0, 500.0));
    gfx.render(&draw);
}
