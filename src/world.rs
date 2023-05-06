use crate::utils::Position;
use tch::Tensor;

#[derive(Debug)]
/// Represents an entity in the world.
///
/// An entity has a position (x, y) and a body, which is a tensor representing its shape.
pub(crate) struct Entity {
    side_length: i64,
    position: Position<f64>,
    body: Tensor,
}

impl Entity {
    pub fn new(side_length: i64, body: Tensor) -> Self {
        Entity {
            side_length,
            position: Position::new(0., 0.),
            body,
        }
    }
}

/// The WorldObject trait is implemented for entities that can be added to the world.
pub trait WorldObject<W> {
    fn position(&self) -> Position<f64>;
    fn set_position(&mut self, x: f64, y: f64);
    fn world_pos(&self) -> Position<i64>;
    fn add_to_map(&self, world: &W);
    fn update(&self, world: &mut W) {}
}

impl<'world> WorldObject<World<'world>> for Entity {
    fn position(&self) -> Position<f64> {
        self.position
    }

    fn set_position(&mut self, x: f64, y: f64) {
        self.position.x = x;
        self.position.y = y;
    }

    fn world_pos(&self) -> Position<i64> {
        self.position.round()
    }

    fn add_to_map(&self, world: &World<'world>) {
        let mut view = world.get_submap(self.world_pos(), self.side_length);
        view += &self.body;
    }
}

/// WorldObject example
pub(crate) struct Square {
    pub entity: Entity,
}

impl Square {
    pub fn new(side_length: i64, world: &World) -> Self {
        Square {
            entity: Entity {
                side_length,
                position: Position::new(0., 0.),
                body: Tensor::ones(
                    &[world.channels, side_length, side_length],
                    (world.val_type, world.device),
                ),
            },
        }
    }
}

impl<'world> WorldObject<World<'world>> for Square {
    fn position(&self) -> Position<f64> {
        self.entity.position()
    }

    fn set_position(&mut self, x: f64, y: f64) {
        self.entity.set_position(x, y);
    }

    fn world_pos(&self) -> Position<i64> {
        self.entity.world_pos()
    }

    fn add_to_map(&self, world: &World) {
        self.entity.add_to_map(world)
    }
}

pub(crate) struct World<'world> {
    map: Tensor,
    height: i64,
    width: i64,
    pub channels: i64,
    decays: Tensor,
    max_values: Tensor,
    pub device: tch::Device,
    val_type: tch::Kind,
    max_field_of_view: i64,
    objects: Vec<&'world dyn WorldObject<World<'world>>>,
    pub random: rand::rngs::StdRng,
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
impl<'world> World<'world> {
    pub fn add_entity(&mut self, object: &'world dyn WorldObject<World<'world>>) {
        self.objects.push(object);
    }

    pub fn new(
        width: i64,
        height: i64,
        channels: i64,
        max_field_of_view: i64,
        decays: &[f64],
        max_values: &[f64],
        device: tch::Device,
        val_type: tch::Kind,
        seed: u64,
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
        tch::manual_seed(seed as i64);
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
            val_type,
            max_field_of_view,
            objects: vec![],
            random: rand::SeedableRng::seed_from_u64(seed),
        }
    }

    /// Given a map with a certain field of view, get_view returns a tensor
    /// that represents the view of the map from the given position.
    ///
    /// The tensor has dimensions (channels, 2 * field_of_view + 1, 2 * field_of_view + 1)
    /// and contains the values of the map in the corresponding positions.
    ///
    /// For example, if the map is (max field of view = 2):
    ///
    /// 0 0 0 0 0 0 0
    /// 0 0 0 0 0 0 0
    /// 0 0 1 1 P 0 0
    /// 0 0 1 2 1 0 0
    /// 0 0 1 1 1 0 0
    /// 0 0 0 0 0 0 0
    /// 0 0 0 0 0 0 0
    /// and the position is P (2,2) with a field of view of 1, then the tensor returned
    /// will be:
    ///
    /// 0 0 0
    /// 1 P 0
    /// 2 1 0
    ///
    /// which corresponds to the view of the map from the position P with a field of view of 1.
    pub fn get_observation(&self, position: Position<i64>, field_of_view: i64) -> Tensor {
        self.map
            .narrow(
                1,
                position.y + self.max_field_of_view - field_of_view,
                2 * field_of_view + 1,
            )
            .narrow(
                2,
                position.x + self.max_field_of_view - field_of_view,
                2 * field_of_view + 1,
            )
    }
    /// Returns a submap of which top left point is in position
    /// The returned tensor has dimensions (channels, side_length, side_length).
    pub fn get_submap(&self, position: Position<i64>, side_lenght: i64) -> Tensor {
        self.map
            .narrow(1, position.y + self.max_field_of_view, side_lenght)
            .narrow(2, position.x + self.max_field_of_view, side_lenght)
    }

    pub fn print(&self) {
        self.map.print()
    }

    /// Update the world
    pub fn update(&mut self) {
        let objects = self.objects.clone(); // To allow to pass a mutable ref of world in objects update
                                            // Update worldObjects
        for object in objects {
            object.update(self)
        }
        // Decay
        self.map *= &self.decays;
        // Add objects to map
        for object in &self.objects {
            object.add_to_map(self)
        }
        // Clamp max values
        self.map.clamp_max_tensor_(&self.max_values); // in place operation, result can safely be ignored
    }
}
