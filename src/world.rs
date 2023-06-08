use std::cell::RefCell;
use std::collections::HashMap;

use crate::graphics::Drawable;
// use auto_delegate::delegate; todo use later for world: https://github.com/elm-register/auto-delegate/issues/2
use notan::draw::Draw;
use notan::math::Vec2;
use notan::prelude::Texture;
use tch::Tensor;

#[derive(Debug)]
/// Represents an entity in the world.
///
/// An entity has a position (x, y) and a body, which is a tensor representing its shape.
pub struct Entity {
    position: Vec2,
    size: Vec2,
    body: Tensor,
    scaled_body: Tensor,
    direction: Vec2,
    pub speed: f32,
}

/// An entity that has a body and can move
impl Entity {
    /// Create a new entity with `base_body`
    pub fn new(body: Tensor) -> Self {
        Entity {
            position: Vec2::new(0., 0.),
            size: Vec2::new(body.size()[0] as f32, body.size()[1] as f32),
            scaled_body: body.copy(),
            body: body,
            direction: Vec2::new(1., 0.),
            speed: 0.,
        }
    }

    /// Sets the size of the entity and resizes its body tensor accordingly.
    ///
    /// # Arguments
    ///
    /// * `width` - The new width of the entity.
    /// * `height` - The new height of the entity.
    ///
    /// # Note
    ///
    /// The body tensor is resized on the CPU and then moved back to the device it was on before.
    pub fn set_size(&mut self, width: f32, height: f32) {
        self.size = Vec2::new(width, height);
        self.scaled_body = tch::vision::image::resize(
            &self.body.to(tch::Device::Cpu),
            width.ceil() as i64,
            height.ceil() as i64,
        )
        .unwrap()
        .to(self.body.device());
    }

    /// Sets the direction of the entity.
    ///
    /// # Arguments
    ///
    /// * `direction` - The new direction of the entity.
    ///
    /// # Note
    ///
    /// The direction is normalized. If the direction is zero, it is set to (0, 0).
    pub fn set_direction(&mut self, direction: Vec2) {
        self.direction = direction.normalize_or_zero();
    }

    /// Returns the direction of the entity.
    pub fn direction(&self) -> Vec2 {
        self.direction
    }
}

impl Drawable for Entity {
    fn draw(&self, _draw: &mut Draw, _: &State) {}
}

/// The WorldObject trait is implemented for entities that can be added to the world.
pub trait WorldObject<W>: Drawable {
    /// Returns the position of the object.
    fn position(&self) -> Vec2;
    /// Sets the position of the object.
    fn set_position(&mut self, x: f32, y: f32);
    /// Add the object to the world map.
    fn add_to_map(&self, world: &W);
    /// Update the object.
    fn update(&mut self, _world: &W, _state: &mut State) {}
    /// Returns the size of the object.
    fn size(&self) -> Vec2;
}

impl WorldObject<World> for Entity {
    fn position(&self) -> Vec2 {
        self.position
    }

    fn set_position(&mut self, x: f32, y: f32) {
        self.position = Vec2::new(x, y);
    }

    fn add_to_map(&self, world: &World) {
        let mut view = world.get_submap(
            self.position(),
            self.size().x.ceil() as i64,
            self.size().y.ceil() as i64,
        );
        view += &self.scaled_body;
    }

    fn size(&self) -> Vec2 {
        self.size
    }
}

/// The world represents the game world.
pub struct World {
    /// The world map is a (channels, height + 2 * max_field_of_view, width + 2 * max_field_of_view) tensor.
    /// Each channel represents a different aspect of the world (RGB, temperature, etc.).
    map: Tensor,
    /// The objects in the world. They will be updated and drawn every time the world is updated.
    objects: RefCell<Vec<Box<dyn WorldObject<World>>>>,
    height: i64,
    width: i64,
    pub channels: i64,
    /// The decay tensor is a (channels) tensor that represents how much each channel decays before entities are added to the world.
    decays: Tensor,
    /// The max_values tensor is a (channels) tensor that represents the maximum value of each channel.
    max_values: Tensor,
    /// The device on which the world map is stored.
    pub device: tch::Device,
    /// The maximum distance that an entity can see in each direction.
    max_field_of_view: i64,
    pub delta_time: f32,
}

/// The State struct represents the current state of the game.
pub struct State {
    /// The random number generator used by the game.
    pub random: rand::rngs::StdRng,
    /// The textures used by the game.
    textures: HashMap<String, Texture>,
}

impl State {
    /// Creates a new State object.
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed for the random number generator.
    /// * `textures` - The textures used by the game.
    pub fn new(seed: u64, textures: HashMap<String, Texture>) -> Self {
        State {
            random: rand::SeedableRng::seed_from_u64(seed),
            textures,
        }
    }

    /// Returns the texture with the given name.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the texture to get.
    pub fn get_texture(&self, name: &str) -> &Texture {
        self.textures.get(name).unwrap()
    }
}

/// Represents the game world.
///
/// The world map is a 3D tensor with dimensions (channels, height + 2 * max_field_of_view, width + 2 * max_field_of_view).
/// Each channel represents a different aspect of the world (e.g. terrain, objects, enemies, etc.).
/// The height and width represent the size of the world, and the max_field_of_view represents the maximum distance
/// that an entity can see in each direction.
///
/// The world contains entities that implement the WorldObject trait, which allows them to be added to the world and interact with it.
/// Entities have a position (x, y) and a body, which is a tensor representing their shape.
///
/// The world also contains decay and max value tensors, which determine how the values in the world decay over time and what their maximum values can be.
///
/// The World struct provides methods for adding entities to the world, getting views of the world from a certain position,
/// and updating the values in the world based on the decay and max value tensors.
impl World {
    /// Creates a new World
    pub fn new(
        width: i64,
        height: i64,
        channels: i64,
        max_field_of_view: i64,
        decays: &[f64],
        max_values: &[f64],
        device: tch::Device,
        val_type: tch::Kind,
        delta_time: f32,
    ) -> Self {
        let map = Tensor::zeros(
            &[
                channels,
                height + 2 * max_field_of_view,
                width + 2 * max_field_of_view,
            ],
            (val_type, device),
        )
        .requires_grad_(false);
        assert_eq!(decays.len() as i64, channels);
        assert_eq!(max_values.len() as i64, channels);
        World {
            map,
            height,
            width,
            channels,
            decays: Tensor::of_slice(decays)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to_device(device)
                .to_kind(val_type),
            max_values: Tensor::of_slice(max_values)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .to_device(device)
                .to_kind(val_type),
            device,
            max_field_of_view,
            objects: RefCell::new(vec![]),
            delta_time,
        }
    }

    /// Adds an entity to the world. The entity will be updated / drawn every time the world is updated / drawn.
    ///
    /// # Arguments
    ///
    /// * `object` - The entity to add to the world.
    pub fn add_entity(&mut self, object: Box<dyn WorldObject<World>>) {
        self.objects.borrow_mut().push(object.into());
    }

    /// Returns the number of world objects in the world.
    pub fn num_objects(&self) -> usize {
        self.objects.borrow().len()
    }

    /// Returns a submap of which top left point is in position
    /// The returned tensor has dimensions (channels, side_length, side_length).
    pub fn get_submap(&self, position: Vec2, width: i64, height: i64) -> Tensor {
        self.map
            .narrow(
                1,
                position.y.round() as i64 + self.max_field_of_view,
                height,
            )
            .narrow(2, position.x.round() as i64 + self.max_field_of_view, width)
    }

    /// Returns a view of the world from the given position and field of view.
    ///
    /// # Arguments
    ///
    /// * `position` - The position of the top left corner of the entity that is getting the view.
    /// * `field_of_view` - The maximum distance that the view can extend in each direction.
    /// * `size` - The size of the entity that is getting the view.
    ///
    /// The returned tensor has dimensions (channels, 2 * field_of_view + size.y.ceil(), 2 * field_of_view + size.x.ceil()).
    /// Changes to the returned tensor will be reflected in the world map.
    pub fn get_observation(&self, position: Vec2, field_of_view: i64, size: Vec2) -> Tensor {
        self.map
            .narrow(
                1,
                position.y.round() as i64 + self.max_field_of_view - field_of_view,
                2 * field_of_view + size.y.ceil() as i64,
            )
            .narrow(
                2,
                position.x.round() as i64 + self.max_field_of_view - field_of_view,
                2 * field_of_view + size.x.ceil() as i64,
            )
    }
    /// Update the position of an entity so that it is within the bounds of the world.
    pub fn bound_position(&self, entity: &mut Entity) {
        // The anchor point of the entity is the top left corner
        if entity.position().x < 0.0 {
            entity.set_position(0., entity.position().y);
        }
        if entity.position().y < 0.0 {
            entity.set_position(entity.position().x, 0.);
        }
        if entity.position().x + entity.size().x.ceil() > self.width as f32 {
            entity.set_position(
                self.width as f32 - entity.size().x.ceil(),
                entity.position().y,
            );
        }
        if entity.position().y + entity.size().y.ceil() > self.height as f32 {
            entity.set_position(
                entity.position().x,
                self.height as f32 - entity.size().y.ceil(),
            );
        }
    }

    /// Print the world map
    pub fn print(&self) {
        self.map.print()
    }

    /// Update the world
    pub fn update(&mut self, state: &mut State) {
        // let objects = self.objects.clone(); // To allow to pass a mutable ref of world in objects update
        // Update worldObjects
        let mut objects = self.objects.borrow_mut();
        for object in objects.iter_mut() {
            object.update(&self, state);
        }

        // Decay
        self.map *= &self.decays;
        // Add objects to map
        for object in objects.iter() {
            object.add_to_map(self)
        }
        // Clamp max values
        let _ = self.map.clamp_max_tensor_(&self.max_values); // in place operation, result can safely be ignored
    }
}

impl Drawable for World {
    fn draw(&self, draw: &mut Draw, state: &State) {
        self.objects
            .borrow()
            .iter()
            .for_each(|object| object.draw(draw, state));
    }
}
