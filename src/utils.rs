use tch::Tensor;

#[derive(Debug, Clone, Copy)]
pub struct Position<I> {
    pub x: I,
    pub y: I,
}

impl<I: std::ops::Add<Output = I>> std::ops::Add for Position<I> {
    type Output = Position<I>;
    fn add(self, rhs: Self) -> Self::Output {
        Position {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<I: std::ops::Mul<Output = I> + Copy> std::ops::Mul<I> for Position<I> {
    type Output = Position<I>;
    fn mul(self, rhs: I) -> Self::Output {
        Position {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<I: std::ops::Add<Output = I> + Copy> std::ops::Add<I> for Position<I> {
    type Output = Position<I>;
    fn add(self, rhs: I) -> Self::Output {
        Position {
            x: self.x + rhs,
            y: self.y + rhs,
        }
    }
}

impl<I: std::ops::Div<Output = I> + Copy> std::ops::Div<I> for Position<I> {
    type Output = Position<I>;
    fn div(self, rhs: I) -> Self::Output {
        Position {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
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

impl Position<f32> {
    pub fn normalize(&self) -> Position<f32> {
        let norm = (self.x.powi(2) + self.y.powi(2)).sqrt();
        Position {
            x: self.x,
            y: self.y,
        } / norm
    }

    pub fn round(&self) -> Position<i64> {
        Position {
            x: self.x.round() as i64,
            y: self.y.round() as i64,
        }
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

pub fn sample_index(weights: &Tensor) -> i64 {
    let indices = weights.multinomial(1, true);
    indices.int64_value(&[0])
}

pub trait EnumFromRepr
where
    Self: Sized,
{
    fn from_repr(discriminant: usize) -> Option<Self>;
}
