use crate::constants::{graphics_c::YAAL_SPRITE, yaal_c::*};
use crate::graphics::sprite_path;
use crate::graphics::Drawable;
use crate::utils;
use crate::world;
use crate::world::Entity;
use crate::world::State;
use delegate::delegate;
use notan::draw::Draw;
use notan::draw::DrawImages;
use notan::math::Vec2;
use rand::rngs::StdRng;
use rand::Rng;
use strum_macros::{EnumCount, FromRepr};
use tch::IndexOp;
use tch::{Kind, Tensor};
use utils::EnumFromRepr;
use world::World;
use world::WorldObject;

/// Trait that allows to create a new object with random values
pub trait RandomInit {
    fn new_random(world: &World, state: &mut State) -> Self;
}

/// Trait for objects that can move
trait MovingObject {
    /// Returns the current position of the object
    fn position(&self) -> Vec2;
    /// Sets the position of the object to the given coordinates
    fn set_position(&mut self, x: f32, y: f32);
    /// Returns the direction of the object's movement
    fn direction(&self) -> Vec2;
    /// Returns the speed of the object's movement
    fn speed(&self) -> f32;
    /// Updates the position of the object based on its direction and speed
    fn update_pos(&mut self, delta: f32) {
        let new_pos = self.position() + self.direction() * self.speed() * delta;
        self.set_position(new_pos.x, new_pos.y);
    }
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
    /// Tensor of shape [world_channels] that contains the weights for the scalar product
    direction_weights: Tensor,
    /// Tensor of shape [world_channels]
    speed_weights: Tensor,
    /// Bias for the scalar product
    speed_bias: f32,
    /// Tensor of shape [world_channels, nb_actionss]
    decision_weights: Tensor,
    /// Bias for the scalar product [nb_actions]
    decision_bias: Tensor,
    /// The type of decision sampling used to choose an action
    decision_sampling: DecisionSampling,
    /// The distance penalty function for the direction
    direction_distance_penalty: DistancePenalty,
    /// The distance penalty function for the speed
    speed_distance_penalty: DistancePenalty,
    /// The distance penalty function for the decision
    decision_distance_penalty: DistancePenalty,
    /// The norm of the random vector used to add noise to the direction
    rand_direction_norm: f32,
}

#[derive(Debug)]
/// A decision made by a Yaal
struct YaalDecision {
    action: YaalAction,
    direction: Vec2,
    speed_factor: f32,
}
impl YaalDecision {
    /// Apply the decision to the Yaal
    fn apply(&self, yaal: &mut Yaal) {
        // TODO: Implement actions
        match self.action {
            YaalAction::Attack => (),
            YaalAction::Reproduce => (),
            YaalAction::Nop => (),
        }
        yaal.entity.set_direction(self.direction);
        yaal.entity.speed = yaal.genome.max_speed * self.speed_factor;
    }
}

impl YaalVectorBrain {
    /// Get the preferred direction of the Yaal
    fn get_direction(&self, input_view: &Tensor, rand_norm: f32, random: &mut StdRng) -> Vec2 {
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
        // x[i,j] = i
        let x = &xy[0];
        // y[i,j] = j
        let y = &xy[1];
        let dir_x = ((x - center) * &eval).mean(Kind::Float);
        let dir_y = ((y - center) * &eval).mean(Kind::Float);
        let rand_dir = Vec2::new(random.gen_range(-1. ..1.), random.gen_range(-1. ..1.))
            .normalize()
            * rand_norm;
        let raw_dir = Vec2::new(dir_x.into(), dir_y.into()) + rand_dir;
        raw_dir.normalize_or_zero()
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
        let logits = eval.mean_dim(Some([0].as_slice()), false, Kind::Float) + &self.decision_bias;
        YaalAction::from_index(self.decision_sampling.logits_to_index(&logits))
    }

    /// Get the preferred speed factor of the Yaal
    fn get_speed_factor(&self, input_view: &Tensor) -> f32 {
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
        let eval: f32 = eval.mean(Kind::Float).into();
        utils::sigmoid(eval + self.speed_bias)
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
    max_speed: f32,
    field_of_view: i64,
    max_size: f32,
    init_size: f32,
}

#[derive(Debug)]
/// The state of a Yaal, can be used as input
struct YaalState {
    health: f64,
    max_health: f64,
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
    sprite: String,
}

// #### Random init ####
impl<T> RandomInit for T
where
    T: EnumFromRepr + strum::EnumCount,
{
    fn new_random(_world: &World, state: &mut State) -> Self {
        let repr = state.random.gen_range(0..T::COUNT);
        T::from_repr(repr).unwrap()
    }
}

impl RandomInit for DistancePenalty {
    fn new_random(world: &World, state: &mut State) -> Self {
        DistancePenalty {
            weight: state.random.gen_range(0. ..1.),
            ptype: PenaltyType::new_random(world, state),
        }
    }
}

impl RandomInit for YaalVectorBrain {
    fn new_random(world: &World, state: &mut State) -> Self {
        let device = world.device;
        let direction_weights = Tensor::randn(&[world.channels], (Kind::Float, device));
        let speed_weights = Tensor::randn(&[world.channels], (Kind::Float, device));
        let decision_weights = Tensor::randn(
            &[world.channels, YaalAction::NB_ACTIONS],
            (Kind::Float, device),
        );
        let decision_sampling = DecisionSampling::new_random(world, state);
        let direction_distance_penalty = DistancePenalty::new_random(world, state);
        let speed_distance_penalty = DistancePenalty::new_random(world, state);
        let decision_distance_penalty = DistancePenalty::new_random(world, state);
        let rand_direction_norm = state.random.gen_range(0. ..1.);
        YaalVectorBrain {
            device,
            direction_weights,
            speed_weights,
            speed_bias: Tensor::randn(&[], (Kind::Float, device)).into(),
            decision_weights,
            decision_bias: Tensor::randn(&[YaalAction::NB_ACTIONS], (Kind::Float, device)),
            decision_sampling,
            direction_distance_penalty,
            speed_distance_penalty,
            decision_distance_penalty,
            rand_direction_norm,
        }
    }
}

impl RandomInit for YaalGenome {
    fn new_random(world: &World, state: &mut State) -> YaalGenome {
        let max_speed = state.random.gen_range(0. ..MAX_SPEED);
        let field_of_view = state.random.gen_range(0..=MAX_FOV);
        let max_size = state.random.gen_range(MIN_SIZE..=MAX_SIZE);
        let init_size = state.random.gen_range(MIN_SIZE..=max_size);
        let brain = YaalVectorBrain::new_random(world, state);
        YaalGenome {
            brain,
            max_speed,
            field_of_view,
            init_size,
            max_size,
        }
    }
}

impl YaalState {
    /// TODO: Take into account the genome
    fn new(_genome: &YaalGenome) -> YaalState {
        YaalState {
            health: 1.,
            max_health: 1.,
            energy: 1.,
            max_energy: 1.,
            age: 0.,
        }
    }
}

impl RandomInit for Yaal {
    fn new_random(world: &World, state: &mut State) -> Yaal {
        let genome = YaalGenome::new_random(world, state);
        let yaal_state = YaalState::new(&genome);
        let rgb = tch::vision::image::load(sprite_path(YAAL_SPRITE))
            .unwrap()
            .to(world.device);
        let blank = Tensor::zeros(
            &[world.channels - 3, rgb.size()[1], rgb.size()[2]],
            (Kind::Float, world.device),
        );
        let body = Tensor::cat(&[rgb, blank], 0);
        let _ = body.i(BODY_CHANNEL).fill_(1.);
        let mut entity = Entity::new(body);
        entity.set_size(genome.init_size, genome.init_size);
        Yaal {
            internal_state: yaal_state,
            entity,
            genome,
            sprite: YAAL_SPRITE.to_string(),
        }
    }
}

impl Yaal {
    pub fn spawn(mut self, x: f32, y: f32, world: &mut World) {
        self.entity.set_position(x, y);
        world.bound_position(&mut self.entity);
        world.add_entity(Box::new(self));
    }
}

impl MovingObject for Yaal {
    delegate! {
        to self.entity {
            fn position(&self) -> Vec2;
            fn set_position(&mut self, x: f32, y: f32);
            fn direction(&self) -> Vec2;
        }
    }
    fn speed(&self) -> f32 {
        self.entity.speed
    }
}

// Allows Yaal to be used as a WorldObject (therefore to be added and interact with the world)
impl WorldObject<World> for Yaal {
    delegate! {
        to self.entity {
            fn position(&self) -> Vec2;
            fn set_position(&mut self, x: f32, y: f32);
            fn add_to_map(&self, world: &World);
            fn size(&self) -> Vec2;
        }
    }
    fn update(&mut self, world: &World, state: &mut State) {
        let sensory_input = world.get_observation(
            self.entity.position(),
            self.genome.field_of_view,
            self.entity.size(),
        );
        let output = self.genome.brain.evaluate(sensory_input, &mut state.random);
        output.apply(self);
        self.update_pos(world.delta_time);
        world.bound_position(&mut self.entity);
        // TODO use output to update the Yaal
        println!("Yaal action: {:#?}", output);
    }
}

impl Drawable for Yaal {
    fn draw(&self, draw: &mut Draw, state: &State) {
        let size = self.size();
        draw.image(state.get_texture(&self.sprite))
            .position(self.entity.position().x, self.entity.position().y)
            .size(size.x, size.y);
    }
}
