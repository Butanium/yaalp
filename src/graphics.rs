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

pub fn image_texture(image: &str, gfx: &mut Graphics) -> Texture {
    let path = std::path::Path::new("assets").join("sprites").join(image);
    let mut buffer = Vec::new();
    let mut file = File::open(path).unwrap();
    file.read_to_end(&mut buffer).unwrap();
    gfx.create_texture().from_image(&buffer).build().unwrap()
}
