use crate::constants;
use crate::utils;
use crate::world;
use rand::rngs::StdRng;
use rand::Rng;
use strum_macros::{EnumCount, FromRepr};
use tch::IndexOp;
use tch::{Kind, Tensor};
use utils::EnumFromRepr;
use utils::Position;
use world::World;
use world::WorldObject;

/// Trait that allows to create a new object with random values
pub trait RandomInit<W> {
    fn new_random(world: &mut W) -> Self;
}

/// Trait for a creature's brain
trait Brain<Input, Output> {
    fn evaluate(&self, input: Input, random: &mut StdRng) -> Output;
}

#[derive(FromRepr, EnumCount, Debug, Clone, Copy)]
/// The type of penalty function used to compute the distance penalty
enum PenaltyType {
    Linear,
    Quadratic,
    Exponential,
}
// Useful for RandomInit
impl EnumFromRepr for PenaltyType {
    fn from_repr(discriminant: usize) -> Option<Self> {
        PenaltyType::from_repr(discriminant)
    }
}

#[derive(Debug, Clone, Copy)]
/// Weighted distance penalty
struct DistancePenalty {
    weight: f64,
    ptype: PenaltyType,
}

impl DistancePenalty {
    /// Computes the distance penalty function for a given distance
    fn compute_penalty(self, dist: &Tensor) -> Tensor {
        match self.ptype {
            PenaltyType::Linear => 1. / (1. + self.weight.max(0.) * dist),
            PenaltyType::Quadratic => 1. / (1. + self.weight.max(0.) * dist.square()),
            PenaltyType::Exponential => 1. / (1. + self.weight.max(0.) * dist.exp()),
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
        let _ = mask.i((side_length / 2, side_length / 2)).fill_(0.);
        // Set dist[side_length/2, side_length/2] to 1
        let _ = dist.i((side_length / 2, side_length / 2)).fill_(1.);
        let max = mask.max();
        let _ = mask.divide_(&max);
        mask.divide_(&dist)
    }
}

#[derive(Debug, Clone, Copy)]
/// The actions that a Yaal can execute
enum YaalAction {
    Attack,
    Reproduce,
    Nop,
}

#[derive(FromRepr, EnumCount, Debug, Clone, Copy)]
/// The type of decision sampling used to choose an action
enum DecisionSampling {
    Softmax,
    Argmax,
    Sample,
}
// Useful for RandomInit
impl EnumFromRepr for DecisionSampling {
    fn from_repr(discriminant: usize) -> Option<Self> {
        DecisionSampling::from_repr(discriminant)
    }
}

impl DecisionSampling {
    /// Sample an index with weighted probabilities from logits
    fn logits_to_index(self, logits: &Tensor) -> i64 {
        match self {
            DecisionSampling::Argmax => logits.argmax(0, false).into(),
            DecisionSampling::Softmax => utils::sample_index(&logits.softmax(0, Kind::Float)),
            DecisionSampling::Sample => {
                let weights = logits - logits.min() + 1e-6;
                utils::sample_index(&weights)
            }
        }
    }
}
impl YaalAction {
    const NB_ACTIONS: i64 = 3;
    fn from_index(index: i64) -> YaalAction {
        match index {
            0 => YaalAction::Attack,
            1 => YaalAction::Reproduce,
            2 => YaalAction::Nop,
            _ => panic!("Invalid action index"),
        }
    }
}
#[derive(Debug)]
/// A brain which decision are mostly based on scalar products
struct YaalVectorBrain {
    device: tch::Device,
    /// Tensor of shape [world_channels]
    direction_weights: Tensor,
    /// Tensor of shape [world_channels]
    speed_weights: Tensor,
    /// Tensor of shape [world_channels, nb_actionss]
    decision_weights: Tensor,
    decision_sampling: DecisionSampling,
    direction_distance_penalty: DistancePenalty,
    speed_distance_penalty: DistancePenalty,
    decision_distance_penalty: DistancePenalty,
    rand_direction_norm: f64,
}

#[derive(Debug)]
/// A decision made by a Yaal
struct YaalDecision {
    action: YaalAction,
    direction: Position<f64>,
    speed_factor: f64,
}

impl YaalVectorBrain {
    /// Get the preferred direction of the Yaal
    fn get_direction(
        &self,
        input_view: &Tensor,
        rand_norm: f64,
        random: &mut StdRng,
    ) -> Position<f64> {
        let dist_mask = self
            .direction_distance_penalty
            .get_direction_mask(self.device, input_view.size()[1]);
        // View returns a (C, N*M) tensor
        let mut eval = input_view
            .reshape(&[input_view.size()[0], -1])
            .transpose_(0, 1)
            .matmul(&self.direction_weights)
            .reshape(&input_view.size()[1..=2]);
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
        let rand_dir = Position::<f64>::new(random.gen_range(0. ..1.), random.gen_range(0. ..1.))
            .normalize()
            * rand_norm;
        let raw_dir: Position<f64> = Position::<f64>::new(dir_x.into(), dir_y.into()) + rand_dir;
        raw_dir.normalize()
    }

    /// Get the preferred speed of the Yaal
    fn get_action(&self, input_view: &Tensor) -> YaalAction {
        // Shape : (N, M) into (N*M,1)
        let mask = self
            .decision_distance_penalty
            .get_mask(self.device, input_view.size()[1])
            .view(-1)
            .unsqueeze(1);
        // (N*M, C) * (C, nb_actions) = (N*M, nb_actions)
        let mut eval = input_view
            .reshape(&[input_view.size()[0], -1])
            .transpose_(0, 1)
            .matmul(&self.decision_weights);
        eval *= mask;
        let logits = eval.mean_dim(Some([0].as_slice()), false, Kind::Float);
        YaalAction::from_index(self.decision_sampling.logits_to_index(&logits))
    }

    /// Get the preferred speed factor of the Yaal
    fn get_speed_factor(&self, input_view: &Tensor) -> f64 {
        let mask = self
            .speed_distance_penalty
            .get_mask(self.device, input_view.size()[1])
            .view(-1);
        // (N*M, C) * (C,) = (N*M,)
        let mut eval = input_view
            .reshape(&[input_view.size()[0], -1])
            .transpose_(0, 1)
            .matmul(&self.speed_weights);
        eval *= mask;
        utils::sigmoid(eval.mean(Kind::Float).into())
    }
}
impl Brain<Tensor, YaalDecision> for YaalVectorBrain {
    fn evaluate(&self, input_view: Tensor, random: &mut StdRng) -> YaalDecision {
        let direction = self.get_direction(&input_view, self.rand_direction_norm, random);
        let action = self.get_action(&input_view);
        let speed_factor = self.get_speed_factor(&input_view);
        YaalDecision {
            action,
            direction,
            speed_factor,
        }
    }
}
#[derive(Debug)]
/// Contains all the genetic information of a Yaal
struct YaalGenome {
    brain: YaalVectorBrain,
    max_speed: f64,
    field_of_view: i64,
    max_size: i64,
}

#[derive(Debug)]
/// The state of a Yaal, can be used as input
struct YaalState {
    health: f64,
    max_health: f64,
    size: i64,
    energy: f64,
    max_energy: f64,
    age: f64,
}

#[derive(Debug)]
/// A little creature
pub struct Yaal {
    internal_state: YaalState,
    entity: world::Entity,
    genome: YaalGenome,
}

// #### Random init ####

impl<T> RandomInit<World<'_>> for T
where
    T: EnumFromRepr + strum::EnumCount,
{
    fn new_random(world: &mut World) -> Self {
        let repr = world.random.gen_range(0..T::COUNT);
        T::from_repr(repr).unwrap()
    }
}

impl RandomInit<World<'_>> for DistancePenalty {
    fn new_random(world: &mut World) -> Self {
        DistancePenalty {
            weight: world.random.gen_range(0. ..1.),
            ptype: PenaltyType::new_random(world),
        }
    }
}

impl RandomInit<World<'_>> for YaalVectorBrain {
    fn new_random(world: &mut World) -> Self {
        let device = world.device;
        let direction_weights = Tensor::randn(&[world.channels], (Kind::Float, device));
        let speed_weights = Tensor::randn(&[world.channels], (Kind::Float, device));
        let decision_weights = Tensor::randn(
            &[world.channels, YaalAction::NB_ACTIONS],
            (Kind::Float, device),
        );
        let decision_sampling = DecisionSampling::new_random(world);
        let direction_distance_penalty = DistancePenalty::new_random(world);
        let speed_distance_penalty = DistancePenalty::new_random(world);
        let decision_distance_penalty = DistancePenalty::new_random(world);
        let rand_direction_norm = world.random.gen_range(0. ..1.);
        YaalVectorBrain {
            device,
            direction_weights,
            speed_weights,
            decision_weights,
            decision_sampling,
            direction_distance_penalty,
            speed_distance_penalty,
            decision_distance_penalty,
            rand_direction_norm,
        }
    }
}

impl RandomInit<World<'_>> for YaalGenome {
    fn new_random(world: &mut World) -> YaalGenome {
        let max_speed = world.random.gen_range(0. ..1.);
        let field_of_view = world.random.gen_range(0..=constants::MAX_FOV);
        let max_size = world.random.gen_range(1..=constants::MAX_SIZE);
        let brain = YaalVectorBrain::new_random(world);
        YaalGenome {
            brain,
            max_speed,
            field_of_view,
            max_size,
        }
    }
}

impl YaalState {
    /// TODO: Take into account the genome
    fn new(genome: &YaalGenome) -> YaalState {
        YaalState {
            health: 1.,
            max_health: 1.,
            energy: 1.,
            size: 1,
            max_energy: 1.,
            age: 0.,
        }
    }
}

impl RandomInit<World<'_>> for Yaal {
    fn new_random(world: &mut World) -> Yaal {
        let genome = YaalGenome::new_random(world);
        let state = YaalState::new(&genome);
        // Todo proper body init
        let square = world::Square::new(state.size, &world);
        Yaal {
            internal_state: state,
            entity: square.entity,
            genome,
        }
    }
}

// Allows Yaal to be used as a WorldObject (therefore to be added and interact with the world)
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

    fn update(&self, world: &mut World) {
        let sensory_input = world.get_observation(
            (self.entity.position() + (self.internal_state.size as f64 / 2.)).round(),
            self.genome.field_of_view,
            self.internal_state.size,
        );
        println!(
            "Yaal input: {:#?}\nshape: {:#?}",
            sensory_input,
            sensory_input.size()
        );
        let output = self.genome.brain.evaluate(sensory_input, &mut world.random);
        // TODO use output to update the Yaal
        println!("Yaal action: {:#?}", output);
    }
}
