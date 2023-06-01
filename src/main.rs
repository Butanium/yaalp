use graphics::Drawable;
use notan::draw::*;
use notan::prelude::*;
use world::WorldObject;
mod constants;
mod creature;
mod graphics;
mod utils;
mod world;
use crate::creature::RandomInit;
use crate::world::World;
use tch::IndexOp;

const WIDTH: i32 = 500;
const HEIGHT: i32 = 500;

#[derive(AppState)]
struct State<'world> {
    world: World<'world>,
    frame: usize,
}

#[notan_main]
fn main() -> Result<(), String> {
    // let win_config = WindowConfig::new().size(WIDTH, HEIGHT).vsync(true);
    let win_config = WindowConfig::new().set_vsync(true);

    notan::init_with(init)
        .add_config(win_config)
        .add_config(DrawConfig)
        .update(update)
        .draw(draw)
        .build()
}
#[test]
fn smoke_test() {
    for _ in 0..100 {
        main().unwrap();
    }
}
const DELTA_TIME: f32 = 1. / 60.;
fn create_world(gfx: &mut Graphics) -> World {
    let _guard = tch::no_grad_guard(); // disable gradient calculation

    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };
    println!("Device used: {:?}", device);
    let seed = rand::random();
    // RGB World
    World::new(
        WIDTH as i64,
        HEIGHT as i64,
        3,
        10,
        &[0., 0., 0.],
        &[255., 255., 255.],
        device,
        tch::Kind::Float,
        seed,
        gfx,
        DELTA_TIME,
    )
}

fn init<'world>(gfx: &mut Graphics) -> State<'world> {
    State {
        world: create_world(gfx),
        frame: 0,
    }
}

fn update<'world>(app: &mut App, state: &mut State<'world>) {
    let _guard = tch::no_grad_guard(); // disable gradient calculation
    state.world.update();
    state.frame += 1;
    if state.frame % 120 == 0 {
        let mut yaal = creature::Yaal::new_random(&mut state.world);
        yaal.set_position(
            state.world.random.gen_range(50. ..250.),
            state.world.random.gen_range(50. ..250.),
        );
        state.world.add_entity(&mut yaal);
    }
}

fn draw<'world>(gfx: &mut Graphics, state: &mut State<'world>) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);
    state.world.draw(&mut draw);
}
 