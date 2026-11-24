//! # sonix-rs
//!
//! Pure Rust, zero-dependency parser, software synthesizer, and byte-exact golden-tested
//! emulator for the **Cosmigo Sonix Sound System** on the Game Boy Advance.
//!
//! Covers all **14 commercial GBA ROMs** across **3 distinct releases** discovered in a
//! full 3,377-ROM corpus audit:
//! - [`SonixVersion::V109c`] (1 ROM: *Ultimate Brain Games*): earliest release, no GSM
//!   voice codec, accumulator buffer at engine offset `0x358`, `>> 2` packed mix shift,
//!   store mask `0xFFFF_3FFF`, accumulate mask `0x0003_0000`, clamp shift `>> 21`.
//! - [`SonixVersion::V1096Gsm`] (11 ROMs: *Ultimate Card Games* (3 revisions) and eight
//!   2-in-1 game packs): adds the GSM voice codec; switches to `>> 3` packed mix shift,
//!   store mask `0xFFFF_1FFF`, accumulate mask `0x0007_0000`, and clamp shift `>> 20`.
//! - [`SonixVersion::V116Gsm`] (2 ROMs: *Tron 2.0: Killer App* USA & Europe): recompiled
//!   mixer with a 4-byte larger engine header, accumulator buffer at engine offset `0x35C`,
//!   and identical mixer math to `V1096Gsm`.
//!
//! All three releases share the same core mixer architecture: unsigned 8-bit samples, a
//! 64-bit phase accumulator (20-bit integer + fraction), packed `(s * vol) >> shift`
//! dual-sample scaling with store/accumulate mask quirks, volume-envelope decay, loop
//! wrap by repeated loop-length subtraction on phase carry, and a DC-bias saturating
//! clamp.
//!
//! ## Features
//! - **Zero dependencies** in `[dependencies]`, `#![no_std]` compatible, `#![deny(unsafe_code)]`.
//! - **Dual synthesis pipelines**:
//!   - [`AccuracySynth`]: 100% bit-exact emulation of all three Sonix ARM7TDMI releases.
//!   - [`StudioSynth`]: 48,000 Hz `f32` band-limited Lanczos-3 polyphase LUT renderer with
//!     true equal-power stereo panning.
//! - **Byte-exact `libmgba` ARM golden tested**: every output byte and updated voice
//!   phase verified strictly byte-for-byte (`assert_eq!`) across all three releases.
//!
//! ## Quickstart
//! ```rust
//! use cosmigo_sonix::{
//!     AccuracySynth, SonixContext, SonixVersion, SonixVoice, SONIX_BLOCK_SIZE, SONIX_BLOCK_WORDS,
//! };
//!
//! let pcm: &[u8] = &[128, 160, 192, 224, 255, 224, 192, 160, 128, 96, 64, 32, 0, 32, 64, 96];
//! let ctx = SonixContext::default();
//! let mut voice = SonixVoice::new_simple(64, 200, 0x0001_0000);
//! let mut acc = [0u32; SONIX_BLOCK_WORDS];
//! let mut out = [0i8; SONIX_BLOCK_SIZE];
//!
//! let total = AccuracySynth::mix_channel(
//!     SonixVersion::V116Gsm,
//!     &mut acc,
//!     pcm,
//!     0,
//!     &mut voice,
//!     &ctx,
//!     false,
//!     SONIX_BLOCK_SIZE,
//! );
//! AccuracySynth::clamp_block(SonixVersion::V116Gsm, &acc, total as u16, &mut out);
//! assert!(out.iter().any(|&s| s != 0));
//! ```

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
