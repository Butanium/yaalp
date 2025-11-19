use crate::world::Position;
use rand::Rng;
use tch::Tensor;

/// A plant entity that provides food in the environment
pub struct Plant {
    pub position: Position<f32>,
    pub body: Tensor,
}

impl Plant {
    /// Create a new plant with a default body
    pub fn new(num_channels: i64, device: tch::Device, val_type: tch::Kind) -> Self {
        let size = 3i64; // Default plant size
        let body = Self::default_body(size, num_channels, device, val_type);
        Plant {
            position: Position::new(0.0, 0.0),
            body,
        }
    }

    /// Generate default plant body
    fn default_body(
        size: i64,
        num_channels: i64,
        device: tch::Device,
        val_type: tch::Kind,
    ) -> Tensor {
        let mut body = Tensor::zeros(&[num_channels, size, size], (val_type, device));
        // Set channel 3 (assuming it's the plant/food channel) to 1.0
        if num_channels > 3 {
            let _ = body.narrow(0, 3, 1).fill_(1.0);
        }
        body
    }

    /// Set a random position within bounds
    pub fn set_random_position(&mut self, min: &Position<f32>, max: &Position<f32>) {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(min.x()..max.x());
        let y = rng.gen_range(min.y()..max.y());
        self.position = Position::new(x, y);
    }

    /// Get the size of the plant
    pub fn size(&self) -> i64 {
        self.body.size()[1]
    }
}
