//! Error types for Sonix parsing and synthesis.

use core::fmt;

/// Errors returned when parsing Sonix headers, sample descriptors, or mixing audio.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// Input buffer ended before a complete header or payload could be read.
    UnexpectedEof,
    /// Header or sample descriptor failed structural validation.
    InvalidHeader(&'static str),
    /// Loop range exceeded or violated the sample length bounds.
    InvalidLoopRange {
        /// Loop start offset in samples.
        loop_start: u32,
        /// Loop end offset in samples.
        loop_end: u32,
        /// Total PCM sample length in bytes.
        pcm_length: usize,
    },
    /// Sample index was out of range for the sample descriptor table.
    InvalidSample(u32),
    /// Channel index was out of range for the channel table.
    InvalidChannel(usize),
    /// Invalid synthesis parameter.
    InvalidParameter(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "Unexpected end of data while parsing Sonix stream"),
            Self::InvalidHeader(msg) => write!(f, "Invalid Sonix header: {msg}"),
            Self::InvalidLoopRange {
                loop_start,
                loop_end,
                pcm_length,
            } => write!(
                f,
                "Invalid Sonix loop range {loop_start}..{loop_end} for {pcm_length}-byte sample"
            ),
            Self::InvalidSample(id) => write!(f, "Invalid Sonix sample index {id}"),
            Self::InvalidChannel(id) => write!(f, "Invalid Sonix channel index {id}"),
            Self::InvalidParameter(msg) => write!(f, "Invalid Sonix parameter: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Convenience result alias for `sonix`.
pub type Result<T> = core::result::Result<T, Error>;
