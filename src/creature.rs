use crate::utils;
use crate::world;
use rand::Rng;
use tch::IndexOp;
use tch::{Kind, Tensor};
use utils::Position;
use world::World;
use world::WorldObject;

trait Brain<Input, Output> {
    fn evaluate(&self, input: Input, world: &mut World) -> Output;
}

#[derive(Debug, Clone, Copy)]
enum PenaltyType {
    Linear,
    Quadratic,
    Exponential,
}

#[derive(Debug, Clone, Copy)]
struct DistancePenalty {
    weight: f64,
    ptype: PenaltyType,
}

impl DistancePenalty {
    fn compute_penalty(self, dist: &Tensor) -> Tensor {
        match self.ptype {
            PenaltyType::Linear => 1. / (1. + self.weight * dist),
            PenaltyType::Quadratic => 1. / (1. + self.weight * dist.square()),
            PenaltyType::Exponential => 1. / (1. + self.weight * dist.exp()),
        }
    }

    /// Returns a `Tensor` of shape `[side_length, side_length]` with values between 0 and 1.
    /// The values are a_{i,j} = 1/(1 + f dist(i,j)) where dist(i,j) is the distance between
    /// the position (i,j) and the center of the map and f is the distance penalty function.
    /// The a_{i,j} are normalized so that the maximum value is 1.
    fn get_mask(&self, device: tch::Device, side_length: i64) -> Tensor {
        // Create two tensors of shape [side_length, side_length] that contain the x and y coordinates of each position
        let x = Tensor::linspace(
            0.0,
            (side_length - 1) as f64,
            side_length,
            (Kind::Float, device),
        );
        let xy = Tensor::meshgrid(&[&x, &x]);
        let x = &xy[0];
        let y = &xy[1];
        // Compute the distance from the center for each position
        let center = (side_length / 2) as f64;
        let dist = ((x - center).square() + (y - center).square()).sqrt();
        // Compute a_{i,j} = 1/(1 + f dist(i,j))
        let mut mask = self.compute_penalty(&dist);
        let max = mask.max();
        mask.divide_(&max)
    }

    /// Returns a `Tensor` of shape `[side_length, side_length]` with values between 0 and 1.
    /// The values are a_{i,j}/dist(i,j) with a_{i,j} = 1/(1 + f dist(i,j)) where dist(i,j) is the distance between
    /// the position (i,j) and the center of the map and f is the distance penalty function.
    /// a_{side_length/2, side_length/2} is set to 0.
    /// The a_{i,j} are normalized so that the maximum value is 1.
    fn get_direction_mask(&self, device: tch::Device, side_length: i64) -> Tensor {
        // Create two tensors of shape [side_length, side_length] that contain the x and y coordinates of each position
        let x = Tensor::linspace(
            0.0,
            (side_length - 1) as f64,
            side_length,
            (Kind::Float, device),
        );
        let xy = Tensor::meshgrid(&[&x, &x]);
        let x = &xy[0];
        let y = &xy[1];
        // Compute the distance from the center for each position
        let center = (side_length / 2) as f64;
        let dist = ((x - center).square() + (y - center).square()).sqrt();
        // Compute a_{i,j} = 1/(1 + f dist(i,j))
        let mut mask = self.compute_penalty(&dist);
        // Set mask[side_length/2, side_length/2] to 0
        mask.i((side_length / 2, side_length / 2)).fill_(0.);
        let max = mask.max();
        mask.divide_(&max);
        mask.divide_(&dist)
    }
}

enum YaalAction {
    Attack,
    Reproduce,
    Nop,
}

struct YaalVectorBrain {
    device: tch::Device,
    direction_weights: Tensor,
    speed_weights: Tensor,
    decision_weights: Tensor,
    direction_distance_penalty: DistancePenalty,
    speed_distance_penalty: DistancePenalty,
    decision_distance_penalty: DistancePenalty,
    rand_direction_norm: f64,
}

struct YaalGenome {
    brain: YaalVectorBrain,
    max_speed: f64,
    field_of_view: i64,
}

struct YaalDecision {
    action: YaalAction,
    direction: Position<f64>,
    speed_factor: f64,
}

impl YaalVectorBrain {
    fn get_direction(
        &self,
        input_view: &Tensor,
        rand_norm: f64,
        world: &mut World,
    ) -> Position<f64> {
        let dist_mask = self
            .direction_distance_penalty
            .get_mask(self.device, input_view.size()[1]);
        let mut eval = input_view.matmul(&self.direction_weights);
        eval *= dist_mask;
        // Compute the average direction vector weighted by eval
        // The origin of the direction vector is the center of the input view
        let side_length = input_view.size()[1];
        let center = (side_length / 2) as f64;
        let x = Tensor::linspace(
            0.0,
            (side_length - 1) as f64,
            side_length,
            (Kind::Float, self.device),
        );
        let xy = Tensor::meshgrid(&[&x, &x]);
        let x = &xy[0]; // because the mask is divided by dist, this is actually x/direction_norm
                        // The /direction_norm is necessary otherwise longer directions would be favored
        let y = &xy[1];
        let dir_x = ((x * &eval) - center).mean(Kind::Float);
        let dir_y = ((y * &eval) - center).mean(Kind::Float);
        let rand_dir = Position::<f64>::new(
            world.random.gen_range(0. ..1.),
            world.random.gen_range(0. ..1.),
        )
        .normalize()
            * rand_norm;
        (Position::new(dir_x.into(), dir_y.into()) + rand_dir).normalize()
    }

    fn get_action(&self, input_view: &Tensor) -> YaalAction {
        todo!("Implement get_action")
    }

    fn get_speed_factor(&self, input_view: &Tensor) -> f64 {
        todo!("Implement get_speed_factor")
    }
}
impl Brain<Tensor, YaalDecision> for YaalVectorBrain {
    fn evaluate(&self, input_view: Tensor, world: &mut World) -> YaalDecision {
        let direction = self.get_direction(&input_view, self.rand_direction_norm, world);
        let action = self.get_action(&input_view);
        let speed_factor = self.get_speed_factor(&input_view);
        YaalDecision {
            action,
            direction,
            speed_factor,
        }
    }
}

struct YaalState {
    health: f64,
    max_health: f64,
    hunger: f64,
    max_hunger: f64,
    age: f64,
}

pub struct Yaal {
    internal_states: YaalState,
    entity: world::Entity,
}

impl<'world> WorldObject<World<'world>> for Yaal {
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
