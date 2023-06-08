use graphics::sprite_textures;
use graphics::Drawable;
use notan::draw::*;
use notan::prelude::*;
use world::State;
mod constants;
mod creature;
mod graphics;
mod utils;
mod world;
use crate::creature::RandomInit;
use crate::world::World;

const WIDTH: u32 = 700;
const HEIGHT: u32 = 700;

#[derive(AppState)]
struct GameState {
    world: World,
    world_state: State,
    frame: usize,
}

#[notan_main]
fn main() -> Result<(), String> {
    // let win_config = WindowConfig::new().size(WIDTH, HEIGHT).vsync(true);
    let win_config = WindowConfig::new().set_size(WIDTH, HEIGHT).set_vsync(true);
    let _guard = tch::no_grad_guard(); // disable gradient calculation

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
fn create_world() -> World {
    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };
    println!("Device used: {:?}", device);

    // RGB World
    World::new(
        WIDTH as i64,
        HEIGHT as i64,
        3,
        constants::MAX_FOV,
        &[0., 0., 0.],
        &[255., 255., 255.],
        device,
        tch::Kind::Float,
        DELTA_TIME,
    )
}

fn init(gfx: &mut Graphics) -> GameState {
    let seed = rand::random();
    tch::manual_seed(seed as i64);
    GameState {
        world: create_world(),
        frame: 0,
        world_state: State::new(seed, sprite_textures(gfx)),
    }
}

fn update(state: &mut GameState) {
    let _guard = tch::no_grad_guard(); // disable gradient calculation
    state.world.update(&mut state.world_state);
    state.frame += 1;
    if state.frame % 120 == 0 {
        let yaal = creature::Yaal::new_random(&state.world, &mut state.world_state);
        yaal.spawn(
            state.world_state.random.gen_range(0. ..WIDTH as f32),
            state.world_state.random.gen_range(50. ..HEIGHT as f32),
            &mut state.world,
        );
    }
}

fn draw(gfx: &mut Graphics, state: &mut GameState) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);
    state.world.draw(&mut draw, &state.world_state);
    gfx.render(&draw);
}
