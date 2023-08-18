/// World specs
pub mod world_c {
    pub const NUM_CHANNELS: i64 = 5;
    // TODO: Rename last 2 channels
    pub const CHANNEL_NAMES: &[&str] = &["Red", "Green", "Blue", "Food", "Animals"];
    // TODO: balance this
    pub const DECAYS: &[f32] = &[0., 0., 0., 0.9, 0.9];
    pub const MAX_VALUES: &[f32] = &[255., 255., 255., 10., 10.];
}

/// Yaal specs:
pub mod yaal_c {
    // TODO: balance all this
    pub const MAX_SIZE: f32 = 30.;
    pub const MIN_SIZE: f32 = 5.;
    pub const MAX_SPEED: f32 = 10.;
    pub const MAX_FOV: i64 = 30;
    pub const BODY_CHANNEL: i64 = 4;
}

/// Sim specs
pub mod graphics_c {
    pub const YAAL_SPRITE: &str = "square.png";
    pub const BACKGROUND_SPRITE: &str = "grass.png";
    pub const SPRITES: &[&str] = &[YAAL_SPRITE, BACKGROUND_SPRITE];
}
