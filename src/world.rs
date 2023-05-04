use tch::Tensor;

#[derive(Debug)]
struct Creature {
    side_length: i64,
    x: f64,
    y: f64,
    body: Tensor,
}

#[derive(Debug)]
pub(crate) struct World {
    map: Tensor,
    height: i64,
    width: i64,
    channels: i64,
    decays: Tensor,
    max_values: Tensor,
    device: tch::Device,
    val_type: tch::Kind,
    max_field_of_view: i64,
    creatures: Vec<Creature>,
}

impl Creature {
    fn position(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    fn world_pos(&self) -> (i64, i64) {
        (self.x.round() as i64, self.y.round() as i64)
    }

    fn add_to_map(&self, world: &World) {
        let mut view = world.get_view(self.world_pos(), self.side_length);
        view += &self.body;
    }
}

impl World {
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
        let map = Tensor::ones(
            // todo zeros
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
            creatures: vec![],
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
    pub fn get_view(&self, position: (i64, i64), field_of_view: i64) -> Tensor {
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

    pub fn print(&self) {
        self.map.print()
    }

    pub fn update(&mut self) {
        self.map *= &self.decays;
        for creature in &self.creatures {
            creature.add_to_map(self)
        }
        self.map.clamp_max_tensor_(&self.max_values); // in place operation
    }
}
