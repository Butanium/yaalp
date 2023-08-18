// implement the functions usefull for evolution here

trait Mutable {
    fn mutation(&mut self, mutation_rate: f32);
}

trait Crossover {
    fn crossover(&self, other: &Self) -> Self;
}

trait Fitness {
    fn fitness(&self) -> f32;
}

trait Evolution {
    fn evolve(&mut self, population: &Vec<Self>, mutation_rate: f32);
}

trait Selection {
    fn select(&self, population: &Vec<Self>) -> Self;
}

trait Reproduction {
    fn reproduce(&self, other: &Self) -> Self;
}

trait Population {
    fn population(&self) -> Vec<Self>;
}

trait GeneticAlgorithm {
    fn genetic_algorithm(&mut self, population: &Vec<Self>, mutation_rate: f32);
}

impl 