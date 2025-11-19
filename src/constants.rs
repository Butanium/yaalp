/// Constants for the simulation
pub mod constants {
    pub const DELTA_T: f32 = 1.0;
    pub const EPSILON: f32 = 1e-6;
    pub const PHYSICS_EPSILON: f32 = 1e-2;

    pub mod yaal {
        pub const MIN_SPEED: f32 = 0.1;
        pub const MAX_SPEED: f32 = 1.0;
        pub const MIN_FIELD_OF_VIEW: i64 = 1;
        pub const MAX_FIELD_OF_VIEW: i64 = 3;
        pub const MIN_SIZE: i64 = 9;
        pub const MAX_SIZE: i64 = 9;
    }

    pub mod environment {
        use super::yaal;
        pub const FILTER_SIZE: i64 = 7;
        pub const SHARED_SIZE: i64 = yaal::MAX_FIELD_OF_VIEW + yaal::MAX_SIZE / 2 + 1;
    }
}
