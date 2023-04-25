use notan::prelude::*;
use notan::draw::*;
use tch::Tensor;

#[notan_main]
fn main() -> Result<(), String> {
    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
    notan::init().draw(draw)
        .add_config(DrawConfig)
        .build()
}

fn draw(gfx: &mut Graphics) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);
    draw.triangle((400.0, 100.0), (100.0, 500.0), (700.0, 500.0));
    gfx.render(&draw);
}
