use tch::Tensor;

#[derive(Debug)]
/// Represents an entity in the world.
///
/// An entity has a position (x, y) and a body, which is a tensor representing its shape.
pub(crate) struct Entity {
    side_length: i64,
    x: f64,
    y: f64,
    body: Tensor,
}

/// The WorldObject trait is implemented for entities that can be added to the world.
pub trait WorldObject<W> {
    fn position(&self) -> (f64, f64);
    fn set_position(&mut self, x: f64, y: f64);
    fn world_pos(&self) -> (i64, i64);
    fn add_to_map(&self, world: &W);
}

impl<'world> WorldObject<World<'world>> for Entity {
    fn position(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    fn set_position(&mut self, x: f64, y: f64) {
        self.x = x;
        self.y = y;
    }

    fn world_pos(&self) -> (i64, i64) {
        (self.x.round() as i64, self.y.round() as i64)
    }

    fn add_to_map(&self, world: &World<'world>) {
        let mut view = world.get_submap(self.world_pos(), self.side_length);
        view += &self.body;
    }
}

/// WorldObject example
pub(crate) struct Square(Entity);

impl Square {
    pub fn new(side_length: i64, world: &World) -> Self {
        Square(Entity {
            side_length,
            x: 0.0,
            y: 0.0,
            body: Tensor::ones(
                &[world.channels, side_length, side_length],
                (world.val_type, world.device),
            ),
        })
    }
}

impl<'world> WorldObject<World<'world>> for Square {
    fn position(&self) -> (f64, f64) {
        self.0.position()
    }

    fn set_position(&mut self, x: f64, y: f64) {
        self.0.set_position(x, y);
    }

    fn world_pos(&self) -> (i64, i64) {
        self.0.world_pos()
    }

    fn add_to_map(&self, world: &World) {
        self.0.add_to_map(world)
    }
}

pub(crate) struct World<'world> {
    map: Tensor,
    height: i64,
    width: i64,
    channels: i64,
    decays: Tensor,
    max_values: Tensor,
    device: tch::Device,
    val_type: tch::Kind,
    max_field_of_view: i64,
    entities: Vec<&'world dyn WorldObject<World<'world>>>,
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
        self.entities.push(object);
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
    ) -> Self {
        let map = Tensor::zeros(
            &[
                channels,
                height + 2 * max_field_of_view,
                width + 2 * max_field_of_view,
            ],
            (val_type, device),
        );
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
            entities: vec![],
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
    pub fn get_observation(&self, position: (i64, i64), field_of_view: i64) -> Tensor {
        self.map
            .narrow(
                1,
                position.1 + self.max_field_of_view - field_of_view,
                2 * field_of_view + 1,
            )
            .narrow(
                2,
                position.0 + self.max_field_of_view - field_of_view,
                2 * field_of_view + 1,
            )
    }
    /// Returns a submap of which top left point is in position
    /// The returned tensor has dimensions (channels, side_length, side_length).
    pub fn get_submap(&self, position: (i64, i64), side_lenght: i64) -> Tensor {
        self.map
            .narrow(1, position.1 + self.max_field_of_view, side_lenght)
            .narrow(2, position.0 + self.max_field_of_view, side_lenght)
    }

    pub fn print(&self) {
        self.map.print()
    }

    /// Update the world
    pub fn update(&mut self) {
        self.map *= &self.decays;
        // todo: update creatures
        for creature in &self.entities {
            creature.add_to_map(self)
        }
        self.map.clamp_max_tensor_(&self.max_values); // in place operation, result can safely be ignored
    }
}
