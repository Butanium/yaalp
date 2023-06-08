use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use notan::draw::{Draw, DrawImages};
use notan::prelude::{Graphics, Texture};

use crate::world::State;

const ASSETS_PATH: &str = "assets";
const SPRITES_PATH: &str = "sprites";
const SPRITES: &'static [&str] = &[
    crate::constants::YAAL_SPRITE,
    crate::constants::BACKGROUND_SPRITE,
];

pub trait Drawable {
    fn draw(&self, draw: &mut Draw, state: &State);
}

pub trait TextureSprite: Drawable {
    fn position(&self) -> (f32, f32);
    fn size(&self) -> (f32, f32);
    fn texture(&self) -> &Texture;
    fn draw(&self, draw: &mut Draw) {
        draw.image(self.texture())
            .position(self.position().0, self.position().1)
            .size(self.size().0, self.size().1);
    }
}

pub fn sprite_path(sprite: &str) -> std::path::PathBuf {
    std::path::Path::new("src")
        .join(ASSETS_PATH)
        .join(SPRITES_PATH)
        .join(sprite)
}
pub fn sprite_textures(gfx: &mut Graphics) -> HashMap<String, Texture> {
    SPRITES
        .iter()
        .map(|sprite| (sprite.to_string(), image_texture(sprite_path(sprite), gfx)))
        .collect()
}

pub fn image_texture(path: std::path::PathBuf, gfx: &mut Graphics) -> Texture {
    let mut buffer = Vec::new();
    println!("Loading image: {:?}", path);
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();
    gfx.create_texture().from_image(&buffer).build().unwrap()
}
