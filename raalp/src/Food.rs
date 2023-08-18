// Implementation of the food.
// The food is an entity and has interactions with the Yaals (the creatures).

// The food can move if pushed by the Yaals, but is constantly slowed down by friction with the environment.

//The food can also release pheromones, which would be further sensed by the yaals.

// TODO : The world should have passive diffusion of pheromones.

/*
* List of interactions with Yaals :
* - The food can be eaten by the Yaals (take damage and die)
* - The food can be pushed by the Yaals (move)
* - The food can damage the Yaals (upon being eaten or collision)
* - The food can slow down (at each frame, depending on the delta time of course)
*/

// Think to add respective functions to the Yaals so that they also interact with the food.

use crate::creature::Yaal;
use crate::creature::RandomInit;
use crate::world::Entity;

use notan::math::Vec2;

use tch::Tensor;

struct FoodState {
    dead: bool,
    health: f32,
    max_health: f32,
    // TODO : this should be in the world.
    friction: f32,
    damage: f32,    // If the food is a cactus for example it should damage the Yaals.
    }

pub struct Food {
    internal_state: FoodState,
    entity: world::Entity,
    sprite: String,
    }

impl FoodState {
    fn new() -> FoodState {
        FoodState {
            dead: false,
            health: 100.,
            max_health: 100.,
        }
    }
}

impl RandomInit for Food {
    fn new_random(world: &World, state: &mut State) -> Food {
        let food_state = FoodState::new();
        let rgb = tch::vision::image::load(sprite_path(FOOD_SPRITE))
            .unwrap()
            .to(world.device);
        // Careful : Food should NOT have initial non zero values in odor related channels ! The odor will spread naturally afterwards.
        let blank = Tensor::zeros(
            &[world.channels - 3, rgb.size()[1], rgb.size()[2]],
            (Kind::Float, world.device),
        );
        let body = Tensor::cat(&[rgb, blank], 0);
        // TODO : attention au body, rajouter les phéromones uniquement sur les channels qu'il faut.
        let _ = body.i(BODY_CHANNEL).fill_(1.);

        // TODO : this is a cc of the Yaal's init, but I don't think the position is randomly initialized. Verify that.
        let entity = Entity::new(body);
        
        Food {
            internal_state: food_state,
            entity,
            sprite: FOOD_SPRITE.to_string(),
        }
    }
}

impl WorldObject<World> for Food {
    pub fn update(&mut self, world: &World, delta_time: f32) {
        // This function is called at each frame.
        // updates the position and then reduce the speed by the friction coefficient.

        self.entity.position += this.entity.speed * delta_time * this.entity.direction ;
        self.entity.speed -= this.friction * delta_time;
    }
}

impl Food {
    pub fn pushed(&mut self, yaal: &mut Yaal) {
        // This function is called when the food is pushed by a Yaal.
        // The food is pushed in the direction of the Yaal's movement.
        // The food is pushed with a force proportional to the Yaal's speed.

        // Direction is the same as the Yaal's direction.
        // Speed is half the Yaal's speed. (arbitrary. A more advanced physics model could be implemented later)

        // TODO : Implement the function.
    }

    pub fn eaten(&mut self,, yaal: &mut Yaal, damage: f32) -> f32 {
        // take damage and eventually die. Also call hurt() on the Yaal.

        quantity = min(damage, self.health);
        self.health -= damage;
        if self.health <= 0. {
            self.dead = true;
        }

        self.hurt(yaal, self.damage);

        return quantity;
    }

    pub fn hurt(&mut self, yaal: &mut Yaal, damage: f32) {
        // inflict damage to the Yaal.
        yaal.take_damage(damage);
    }
}

impl Drawable for Food {
    fn draw(&self, draw: &mut Draw, state: &State) {
        let size = self.size();
        draw.image(state.get_texture(&self.sprite))
            .position(self.entity.position().x, self.entity.position().y)
            .size(size.x, size.y);
    }
}



/*

Le monde c'est vraiment une grosse matrice ? c'est pas beaucoup trop lourd à faire tourner ?

Ca sert à quoi les pub fn ?
    (copilot :)
    -> C'est pour pouvoir appeler les fonctions depuis l'extérieur du module. (en gros, c'est pour pouvoir les utiliser dans le main.rs)

Pour la fonction draw par exemple je peux la faire directement comme ça ou je dois implémenter le trait drawable ?
    (copilot :)
    -> Tu peux la faire directement comme ça, mais c'est mieux de l'implémenter dans un trait, comme ça tu peux l'appeler sur n'importe quel objet qui implémente le trait.

Les traits c'est sensé permetre de réduire la taille du code quand il y a des fonctions qui sont les mêmes (genre draw pour Yaal et pour Food ce sera la même fonction).
A quoi ça sert d'avoir un trait si je dois l'implémenter pour Yaal et Food ?
    (copilot :)
    -> C'est pour pouvoir appeler la fonction draw sur n'importe quel objet qui implémente le trait. Si tu implémentes la fonction draw sur Yaal et Food, tu pourras appeler draw sur un objet de type Yaal ou Food, mais pas sur un objet de type Creature par exemple. Si tu implémentes la fonction draw dans un trait, tu pourras appeler draw sur n'importe quel objet qui implémente le trait, donc sur un objet de type Creature, Yaal, Food, etc.
    (moi :)
    -> wesh c'est moisi comme explication. Que le trait soit là ou pas je peux implémenter la fonction, ça prendra autant de lignes de code, et je pourrais l'appeler pareil !
    (copilot :)
    -> Oui, mais si tu implémentes la fonction draw dans un trait, tu pourras appeler draw sur n'importe quel objet qui implémente le trait, donc sur un objet de type Creature, Yaal, Food, etc. Si tu implémentes la fonction draw sur Yaal et Food, tu pourras appeler draw sur un objet de type Yaal ou Food, mais pas sur un objet de type Creature par exemple.
    (moi :)
    -> Mais je pourrais aussi appeler draw si je l'implémente directement, sans passer par un trait !!!
    (copilot :)
    -> Oui, mais si tu implémentes la fonction draw dans un trait, tu pourras appeler draw sur n'importe quel objet qui implémente le trait, donc sur un objet de type Creature, Yaal, Food, etc. Si tu implémentes la fonction draw sur Yaal et Food, tu pourras appeler draw sur un objet de type Yaal ou Food, mais pas sur un objet de type Creature par exemple.
    (moi :)
    -> Ok, cpt.
*/