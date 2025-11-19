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
}

impl Yaal {
    /// Create a new Yaal with given genome and position
    pub fn new(position: Position<f32>, genome: YaalGenome, body: Tensor) -> Self {
        Yaal {
            position,
            direction: Position::new(0.0, 0.0),
            genome,
            body,
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

    /// Update the Yaal based on its view of the environment
    pub fn update(&mut self, input_view: &Tensor) {
        let view_size = self.genome.field_of_view * 2 + self.genome.size;
        let decision = self.genome.brain.evaluate(input_view, view_size, view_size);

        // Update position based on decision
        let speed = self.genome.max_speed * decision.speed_factor * constants::DELTA_T;
        self.position = Position::new(
            self.position.x() + decision.direction.x() * speed,
            self.position.y() + decision.direction.y() * speed,
        );
        self.direction = decision.direction;
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
