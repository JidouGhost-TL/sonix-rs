//! # sonix-rs
//!
//! Pure Rust, zero-dependency parser, software synthesizer, and byte-exact golden-tested
//! emulator for the **Cosmigo Sonix Sound System** on the Game Boy Advance.
//!
//! Covers all **14 commercial GBA ROMs** across **3 distinct releases** discovered in a
//! full 3,377-ROM corpus audit:
//! - [`SonixVersion::V109c`] (1 ROM: *Ultimate Brain Games*): earliest release, no GSM
//!   voice codec, accumulator buffer at engine offset `0x358`.
//! - [`SonixVersion::V1096Gsm`] (11 ROMs: *Ultimate Card Games* (3 revisions) and eight
//!   2-in-1 game packs): adds the GSM voice codec; mixer differs from `V109c` only in
//!   register allocation.
//! - [`SonixVersion::V116Gsm`] (2 ROMs: *Tron 2.0: Killer App* USA & Europe): recompiled
//!   mixer with a 4-byte larger engine header, accumulator buffer at engine offset `0x35C`.
//!
//! All three releases share identical mixer math: unsigned 8-bit samples, a 64-bit
//! phase accumulator (20-bit integer + fraction), packed `(s * vol) >> 3` dual-sample
//! scaling with store/accumulate mask quirks, volume-envelope decay, loop wrap by
//! repeated loop-length subtraction on phase carry, and a DC-bias saturating clamp
//! (`clamp((acc >> 4) - total_volume, -128, 127)`).
//!
//! ## Features
//! - **Zero dependencies** in `[dependencies]`, `#![no_std]` compatible, `#![deny(unsafe_code)]`.
//! - **Dual synthesis pipelines**:
//!   - [`AccuracySynth`]: 100% bit-exact emulation of all three Sonix ARM7TDMI releases.
//!   - [`StudioSynth`]: 48,000 Hz `f32` band-limited Lanczos-3 polyphase LUT renderer with
//!     true equal-power stereo panning.
//! - **Byte-exact `libmgba` ARM golden tested**: every output byte and updated voice
//!   phase verified strictly byte-for-byte (`assert_eq!`) across all three releases.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod error;
pub mod freq;
pub mod sample;
pub mod synth;
pub mod version;
pub mod voice;

pub use error::{Error, Result};
pub use freq::{
    effective_volume, freq_to_step, rate_from_timer_reload, GBA_CLOCK_HZ, SONIX_ACCUM_MASK,
    SONIX_BLOCK_SIZE, SONIX_BLOCK_WORDS, SONIX_MAX_CHANNELS, SONIX_PHASE_BIAS, SONIX_STORE_MASK,
};
#[cfg(feature = "alloc")]
pub use sample::SonixSample;
pub use sample::{SonixSampleDesc, SonixSampleView, GBA_ROM_BUS_BASE};
pub use synth::{AccuracySynth, SonixContext, StudioSynth};
pub use version::SonixVersion;
pub use voice::SonixVoice;
