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
use constants::{graphics_c::*, world_c::*, yaal_c::MAX_FOV};

const WIDTH: u32 = 700;
const HEIGHT: u32 = 700;
const DELTA_TIME: f32 = 1. / 60.;

#[derive(AppState)]
struct GameState {
    world: World,
    world_state: State,
    frame: usize,
    font: Font,
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

fn create_world() -> World {
    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    };
    println!("Device used: {:?}", device);
    World::new(
        WIDTH as i64,
        HEIGHT as i64,
        NUM_CHANNELS,
        MAX_FOV,
        DECAYS,
        MAX_VALUES,
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
        font: gfx
            .create_font(include_bytes!("./assets/fonts/Ubuntu-B.ttf"))
            .unwrap(),
    }
}

fn update(app: &mut App, state: &mut GameState) {
    if app.mouse.was_pressed(MouseButton::Left) {
        let (mx, my) = app.mouse.position();
        let yaal = creature::Yaal::new_random(&state.world, &mut state.world_state);
        yaal.spawn(mx, my, &mut state.world);
    }
    let _guard = tch::no_grad_guard(); // disable gradient calculation
    state.world.update(&mut state.world_state);
    state.frame += 1;
}

fn draw(app: &mut App, gfx: &mut Graphics, state: &mut GameState) {
    let mut draw = gfx.create_draw();
    draw.clear(Color::BLACK);
    draw.image(state.world_state.get_texture(BACKGROUND_SPRITE))
        .size(WIDTH as f32, HEIGHT as f32);
    state.world.draw(&mut draw, &state.world_state);
    draw.text(
        &state.font,
        &format!(
            "FPS : {}\nCreatures: {}",
            app.timer.fps().round(),
            state.world.num_objects()
        ),
    )
    .position(10.0, 10.0)
    .size(24.0);
    gfx.render(&draw);
}
