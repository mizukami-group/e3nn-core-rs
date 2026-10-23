use thiserror::Error;

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CoreError {
    #[error("shape mismatch in {op}: expected {expected}, got {actual}")]
    ShapeMismatch {
        op: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("invalid irreps: {0}")]
    InvalidIrreps(String),

    #[error("invalid tensor product: {0}")]
    InvalidTensorProduct(String),

    #[error("invalid fixed MLP: {0}")]
    InvalidMlp(String),

    #[error("invalid radial primitive input: {0}")]
    InvalidRadial(String),

    #[error("invalid normalization primitive input: {0}")]
    InvalidNormalization(String),

    #[error("invalid SO3 primitive input: {0}")]
    InvalidSo3(String),

    #[error("invalid SO2 primitive input: {0}")]
    InvalidSo2(String),

    #[error("invalid scatter primitive input: {0}")]
    InvalidScatter(String),

    #[error("invalid activation primitive input: {0}")]
    InvalidActivation(String),

    #[error("invalid UMA MoLE primitive input: {0}")]
    InvalidUmaMole(String),

    #[error("unsupported feature: {0}")]
    Unsupported(String),
}
