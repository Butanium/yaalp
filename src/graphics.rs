use std::fs::File;
use std::io::Read;

use notan::draw::{Draw, DrawImages};
use notan::prelude::{Graphics, Texture};

pub trait Drawable {
    fn draw(&self, draw: &mut Draw);
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

fn image_texture(image: &str, gfx: &Graphics) -> Texture {
    let path = std::path::Path::new("assets").join("sprites").join(image);
    let mut buffer = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();
    gfx.create_texture().from_image(&buffer).build().unwrap()
}

/// Contains a map<filename, Texture> and a gfx to create textures
pub struct TextureManager<'a> {
    textures: std::collections::HashMap<String, Texture>,
    gfx: &'a mut Graphics,
}

impl<'a> TextureManager<'a> {
    pub fn new(gfx: &mut Graphics) -> Self {
        Self {
            textures: std::collections::HashMap::new(),
            gfx,
        }
    }

    pub fn get(&mut self, image: &str) -> &Texture {
        if !self.textures.contains_key(image) {
            self.textures
                .insert(image.to_string(), image_texture(image, &self.gfx));
        }
        self.textures.get(image).unwrap()
    }
}
