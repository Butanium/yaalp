use crate::constants::constants;
use crate::world::Position;
use rand::Rng;
use tch::Tensor;

/// Decision made by a Yaal
pub struct YaalDecision {
    pub direction: Position<f32>,
    pub speed_factor: f32,
}

/// A simple Multi-Layer Perceptron for Yaal's brain
pub struct YaalMLP {
    /// Weights for computing direction from the input view
    pub direction_weights: Tensor,
}

impl YaalMLP {
    /// Create a random brain with given number of channels
    pub fn random(num_channels: i64, device: tch::Device, val_type: tch::Kind) -> Self {
        let direction_weights = Tensor::randn(&[num_channels], (val_type, device));
        YaalMLP { direction_weights }
    }

    /// Evaluate the brain on the given input view
    pub fn evaluate(&self, input_view: &Tensor, height: i64, width: i64) -> YaalDecision {
        let direction = self.get_direction(input_view, height, width);
        YaalDecision {
            direction,
            speed_factor: 1.0,
        }
    }

    /// Compute direction from input view
    fn get_direction(&self, input_view: &Tensor, height: i64, width: i64) -> Position<f32> {
        // Contract input_view with direction_weights along the channel dimension
        // input_view: (C, H, W)
        // direction_weights: (C,)
        // Result: (H, W) weight map

        let view_shape = input_view.size();
        let channels = view_shape[0];

        // Reshape input_view to (H*W, C) and direction_weights to (C,)
        let reshaped_view = input_view
            .permute(&[1, 2, 0])
            .reshape(&[height * width, channels]);

        // Matrix multiplication: (H*W, C) x (C, 1) = (H*W, 1)
        let weights_col = self.direction_weights.unsqueeze(1);
        let weight_map = reshaped_view.matmul(&weights_col).reshape(&[height, width]);

        // Create direction matrix: directions[i][j] = (i - center, j - center)
        let center_y = height / 2;
        let center_x = width / 2;

        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let weight_data = Vec::<f32>::from(&weight_map);

        for i in 0..height {
            for j in 0..width {
                let idx = (i * width + j) as usize;
                let weight = weight_data[idx];
                let dy = (i - center_y) as f32;
                let dx = (j - center_x) as f32;
                sum_x += dx * weight;
                sum_y += dy * weight;
            }
        }

        let norm = (sum_x * sum_x + sum_y * sum_y).sqrt();
        if norm < constants::EPSILON {
            return Position::new(0.0, 0.0);
        }

        Position::new(sum_x / norm, sum_y / norm)
    }
}

/// The genome of a Yaal containing brain and physical parameters
pub struct YaalGenome {
    pub brain: YaalMLP,
    pub max_speed: f32,
    pub field_of_view: i64,
    pub size: i64,
    pub signature: Vec<f32>,
}

impl YaalGenome {
    /// Create a random genome
    pub fn random(num_channels: i64, device: tch::Device, val_type: tch::Kind) -> Self {
        let mut rng = rand::thread_rng();

        let brain = YaalMLP::random(num_channels, device, val_type);
        let max_speed = rng.gen_range(constants::yaal::MIN_SPEED..constants::yaal::MAX_SPEED);
        let field_of_view =
            rng.gen_range(constants::yaal::MIN_FIELD_OF_VIEW..=constants::yaal::MAX_FIELD_OF_VIEW);
        let size = rng.gen_range(constants::yaal::MIN_SIZE..=constants::yaal::MAX_SIZE);

        // Generate a random signature (3 values for RGB-like identification)
        let signature = vec![rng.gen::<f32>(), rng.gen::<f32>(), rng.gen::<f32>()];

        YaalGenome {
            brain,
            max_speed,
            field_of_view,
            size,
            signature,
        }
    }

    /// Clone genome with small mutations
    pub fn clone_with_mutation(&self) -> Self {
        let mut rng = rand::thread_rng();
        let mutation_rate = 0.1;

        // Mutate brain weights
        let mutated_weights = &self.brain.direction_weights
            + &Tensor::randn_like(&self.brain.direction_weights) * mutation_rate;
        let mutated_brain = YaalMLP {
            direction_weights: mutated_weights,
        };

        // Slightly mutate other parameters
        let max_speed = (self.max_speed + rng.gen_range(-0.05..0.05))
            .clamp(constants::yaal::MIN_SPEED, constants::yaal::MAX_SPEED);

        // Signature inherits from parent with slight variation
        let signature = self
            .signature
            .iter()
            .map(|&v| (v + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0))
            .collect();

        YaalGenome {
            brain: mutated_brain,
            max_speed,
            field_of_view: self.field_of_view, // Keep same FOV
            size: self.size,                   // Keep same size
            signature,
        }
    }

    /// Generate body tensor from genome
    pub fn generate_body(
        &self,
        num_channels: i64,
        device: tch::Device,
        val_type: tch::Kind,
    ) -> Tensor {
        let mut body = Tensor::zeros(&[num_channels, self.size, self.size], (val_type, device));

        // Set channel 0 (signature R) to signature[0]
        if num_channels > 0 && !self.signature.is_empty() {
            let _ = body.narrow(0, 0, 1).fill_(self.signature[0] as f64);
        }
        // Set channel 1 (signature G) to signature[1]
        if num_channels > 1 && self.signature.len() > 1 {
            let _ = body.narrow(0, 1, 1).fill_(self.signature[1] as f64);
        }
        // Set channel 2 (signature B) to signature[2]
        if num_channels > 2 && self.signature.len() > 2 {
            let _ = body.narrow(0, 2, 1).fill_(self.signature[2] as f64);
        }

        body
    }
}

/// A Yaal - the main creature in the simulation
pub struct Yaal {
    pub position: Position<f32>,
    pub direction: Position<f32>,
    pub genome: YaalGenome,
    pub body: Tensor,
    pub energy: f32,
    pub max_energy: f32,
    pub health: f32,
    pub max_health: f32,
    pub age: i64,
}

impl Yaal {
    /// Create a new Yaal with given genome and position
    pub fn new(position: Position<f32>, genome: YaalGenome, body: Tensor) -> Self {
        let size_factor = genome.size as f32;
        let max_energy = 100.0 * size_factor;
        let max_health = 100.0 * size_factor;

        Yaal {
            position,
            direction: Position::new(0.0, 0.0),
            genome,
            body,
            energy: max_energy * 0.8, // Start with 80% energy
            max_energy,
            health: max_health,
            max_health,
            age: 0,
        }
    }

    /// Create a random Yaal
    pub fn random(
        num_channels: i64,
        position: Position<f32>,
        device: tch::Device,
        val_type: tch::Kind,
    ) -> Self {
        let genome = YaalGenome::random(num_channels, device, val_type);
        let body = genome.generate_body(num_channels, device, val_type);
        Self::new(position, genome, body)
    }

    /// Check if the Yaal is alive
    pub fn is_alive(&self) -> bool {
        self.energy > 0.0 && self.health > 0.0
    }

    /// Consume energy for movement and living
    fn consume_energy(&mut self, amount: f32) {
        self.energy = (self.energy - amount).max(0.0);

        // If energy is too low, start losing health
        if self.energy < 10.0 {
            self.health = (self.health - 1.0).max(0.0);
        }
    }

    /// Eat food and gain energy
    pub fn eat(&mut self, food_value: f32) {
        self.energy = (self.energy + food_value).min(self.max_energy);

        // Eating also restores a bit of health
        if self.health < self.max_health {
            self.health = (self.health + food_value * 0.1).min(self.max_health);
        }
    }

    /// Check if Yaal can reproduce
    pub fn can_reproduce(&self) -> bool {
        self.energy > self.max_energy * 0.7 && self.age > 10
    }

    /// Reproduce and create offspring (asexual reproduction with mutation)
    pub fn reproduce(&mut self, num_channels: i64) -> Option<Yaal> {
        if !self.can_reproduce() {
            return None;
        }

        // Reproduction costs energy
        let reproduction_cost = self.max_energy * 0.4;
        self.energy -= reproduction_cost;

        // Create offspring with slightly mutated genome
        let mut offspring_genome = self.genome.clone_with_mutation();
        let offspring_body =
            offspring_genome.generate_body(num_channels, self.body.device(), self.body.kind());

        // Offspring starts near parent
        let mut rng = rand::thread_rng();
        let offset_x = rng.gen_range(-5.0..5.0);
        let offset_y = rng.gen_range(-5.0..5.0);
        let offspring_pos =
            Position::new(self.position.x() + offset_x, self.position.y() + offset_y);

        Some(Yaal::new(offspring_pos, offspring_genome, offspring_body))
    }

    /// Update the Yaal based on its view of the environment
    pub fn update(&mut self, input_view: &Tensor) {
        self.age += 1;

        let view_size = self.genome.field_of_view * 2 + self.genome.size;
        let decision = self.genome.brain.evaluate(input_view, view_size, view_size);

        // Calculate energy cost based on movement and size
        let speed = self.genome.max_speed * decision.speed_factor * constants::DELTA_T;
        let movement_cost = speed * (self.genome.size as f32) * 0.1;
        let living_cost = 0.5; // Base metabolic cost

        self.consume_energy(movement_cost + living_cost);

        // Only move if alive
        if self.is_alive() {
            self.position = Position::new(
                self.position.x() + decision.direction.x() * speed,
                self.position.y() + decision.direction.y() * speed,
            );
            self.direction = decision.direction;
        }
    }

    /// Get the top-left position of the Yaal's body
    pub fn top_left_position(&self) -> Position<f32> {
        let half_size = self.genome.size as f32 / 2.0;
        Position::new(self.position.x() - half_size, self.position.y() - half_size)
    }

    /// Set a random position within bounds
    pub fn set_random_position(&mut self, min: &Position<f32>, max: &Position<f32>) {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(min.x()..max.x());
        let y = rng.gen_range(min.y()..max.y());
        self.position = Position::new(x, y);
    }

    /// Bound the position within min and max
    pub fn bound_position(&mut self, min: &Position<f32>, max: &Position<f32>) {
        let x = self.position.x().clamp(min.x(), max.x());
        let y = self.position.y().clamp(min.y(), max.y());
        self.position = Position::new(x, y);
    }
}
