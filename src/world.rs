use crate::plant::Plant;
use crate::yaal::Yaal;
use tch::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct Position<I> {
    x: I,
    y: I,
}

impl<I> From<(I, I)> for Position<I> {
    fn from(pos: (I, I)) -> Self {
        Position { x: pos.0, y: pos.1 }
    }
}

impl<I> From<Position<I>> for (I, I) {
    fn from(pos: Position<I>) -> Self {
        (pos.x, pos.y)
    }
}

impl<I: Copy> Position<I> {
    pub fn new(x: I, y: I) -> Self {
        Position { x, y }
    }

    pub fn x(&self) -> I {
        self.x
    }

    pub fn y(&self) -> I {
        self.y
    }

    pub fn map<F, J>(&self, f: F) -> Position<J>
    where
        F: Fn(I) -> J,
    {
        Position {
            x: f(self.x),
            y: f(self.y),
        }
    }
}

/// The simulation world
pub struct World {
    pub map: Tensor,
    pub height: i64,
    pub width: i64,
    pub channels: i64,
    pub decays: Tensor,
    pub max_values: Tensor,
    pub device: tch::Device,
    pub val_type: tch::Kind,
    pub max_field_of_view: i64,
    pub yaals: Vec<Yaal>,
    pub plants: Vec<Plant>,
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
            yaals: vec![],
            plants: vec![],
        }
    }

    /// Add a Yaal to the world
    pub fn add_yaal(&mut self, yaal: Yaal) {
        self.yaals.push(yaal);
    }

    /// Add a Plant to the world
    pub fn add_plant(&mut self, plant: Plant) {
        self.plants.push(plant);
    }

    /// Create random Yaals and Plants
    pub fn create_yaals_and_plants(&mut self, num_yaals: i64, num_plants: i64) {
        let min = Position::new(0.0, 0.0);
        let max = Position::new(self.width as f32, self.height as f32);

        for _ in 0..num_yaals {
            let mut yaal = Yaal::random(
                self.channels,
                Position::new(0.0, 0.0),
                self.device,
                self.val_type,
            );
            yaal.set_random_position(&min, &max);
            self.add_yaal(yaal);
        }

        for _ in 0..num_plants {
            let mut plant = Plant::new(self.channels, self.device, self.val_type);
            plant.set_random_position(&min, &max);
            self.add_plant(plant);
        }
    }

    /// Get the view for a Yaal at its position
    pub fn get_view_for_yaal(&self, yaal: &Yaal) -> Tensor {
        let top_left = yaal.top_left_position();
        let view_size = yaal.genome.field_of_view * 2 + yaal.genome.size;

        let y = (top_left.y().round() as i64).max(0).min(self.height);
        let x = (top_left.x().round() as i64).max(0).min(self.width);

        self.map
            .narrow(0, 0, self.channels)
            .narrow(1, y, view_size)
            .narrow(2, x, view_size)
    }

    /// Add a Yaal's body to the map
    fn add_yaal_to_map(&mut self, yaal: &Yaal) {
        let top_left = yaal.top_left_position();
        let y = (top_left.y().round() as i64 + self.max_field_of_view)
            .max(0)
            .min(self.height + 2 * self.max_field_of_view - yaal.genome.size);
        let x = (top_left.x().round() as i64 + self.max_field_of_view)
            .max(0)
            .min(self.width + 2 * self.max_field_of_view - yaal.genome.size);

        let mut submap = self
            .map
            .narrow(0, 0, self.channels)
            .narrow(1, y, yaal.genome.size)
            .narrow(2, x, yaal.genome.size);

        submap += &yaal.body;
    }

    /// Add a Plant's body to the map
    fn add_plant_to_map(&mut self, plant: &Plant) {
        let y = (plant.position.y().round() as i64 + self.max_field_of_view)
            .max(0)
            .min(self.height + 2 * self.max_field_of_view - plant.size());
        let x = (plant.position.x().round() as i64 + self.max_field_of_view)
            .max(0)
            .min(self.width + 2 * self.max_field_of_view - plant.size());

        let mut submap = self
            .map
            .narrow(0, 0, self.channels)
            .narrow(1, y, plant.size())
            .narrow(2, x, plant.size());

        submap += &plant.body;
    }

    pub fn print(&self) {
        println!("World map (channels x height x width):");
        self.map.print();
    }

    /// Perform a simulation step
    pub fn step(&mut self) {
        // 1. Apply decay to the map
        self.map *= &self.decays;

        // 2. Update all Yaals
        for i in 0..self.yaals.len() {
            let view = self.get_view_for_yaal(&self.yaals[i]);
            self.yaals[i].update(&view);

            // Bound position within world
            let min = Position::new(0.0, 0.0);
            let max = Position::new(self.width as f32, self.height as f32);
            self.yaals[i].bound_position(&min, &max);
        }

        // 3. Add all entities to the map
        for yaal in &self.yaals {
            self.add_yaal_to_map(yaal);
        }
        for plant in &self.plants {
            self.add_plant_to_map(plant);
        }

        // 4. Clamp values to max
        let _ = self.map.clamp_max_tensor_(&self.max_values);
    }
}
