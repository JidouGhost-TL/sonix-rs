//! Sonix sample descriptor and zero-allocation unsigned 8-bit PCM sample views.
//!
//! Sonix samples are unsigned 8-bit PCM. The mixer treats the byte as an
//! unsigned magnitude and removes the DC bias in the clamp stage by
//! subtracting the accumulated total channel volume, so a silent sample
//! sits at the byte whose scaled value equals the total volume.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::error::{Error, Result};

/// GBA ROM bus window base address (`0x0800_0000`).
pub const GBA_ROM_BUS_BASE: u32 = 0x0800_0000;

/// Parsed Sonix sample descriptor (the ARM `sl` struct).
///
/// Binary layout (little-endian words as stored in RAM descriptors):
/// - `+0x00`: `base_addend_a: u32` — first sample-address addend.
/// - `+0x04`: `loop_len: u32` — loop length in phase-integer units, subtracted
///   from the phase high word on loop wrap (`0` for one-shot samples).
/// - `+0x0C`: `base_addend_b: u32` — second sample-address addend.
///
/// The ARM sample fetch address is `base_addend_b + engine_bank_base + base_addend_a`
/// plus the biased phase high word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SonixSampleDesc {
    /// First sample-address addend (`+0x00`).
    pub base_addend_a: u32,
    /// Loop length in phase-integer units (`+0x04`); `0` for one-shot samples.
    pub loop_len: u32,
    /// Second sample-address addend (`+0x0C`).
    pub base_addend_b: u32,
}

impl SonixSampleDesc {
    /// Create a descriptor for a one-shot sample.
    #[inline]
    #[must_use]
    pub const fn new_oneshot(base_addend_a: u32, base_addend_b: u32) -> Self {
        Self {
            base_addend_a,
            loop_len: 0,
            base_addend_b,
        }
    }

    /// Create a descriptor for a looped sample.
    #[inline]
    #[must_use]
    pub const fn new_looped(base_addend_a: u32, loop_len: u32, base_addend_b: u32) -> Self {
        Self {
            base_addend_a,
            loop_len,
            base_addend_b,
        }
    }

    /// Returns `true` if this sample loops (`loop_len != 0`).
    #[inline]
    #[must_use]
    pub const fn is_looped(&self) -> bool {
        self.loop_len != 0
    }
}

/// Zero-allocation borrowed view of a Sonix unsigned 8-bit PCM sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SonixSampleView<'a> {
    /// Raw unsigned 8-bit linear PCM bytes.
    pub pcm_bytes: &'a [u8],
    /// Native playback sample rate in Hz.
    pub sample_rate_hz: u32,
    /// MIDI root key (`0..=127`; default `60` = C4).
    pub root_key: u8,
    /// Loop length in samples, or `0` for one-shot samples.
    pub loop_len: u32,
    /// Optional loop start offset in samples.
    pub loop_start: Option<u32>,
    /// Optional loop end offset in samples.
    pub loop_end: Option<u32>,
}

impl<'a> SonixSampleView<'a> {
    /// Construct a `SonixSampleView` directly from raw unsigned 8-bit PCM bytes.
    #[inline]
    #[must_use]
    pub const fn from_pcm_bytes(
        sample_rate_hz: u32,
        root_key: u8,
        loop_start: Option<u32>,
        loop_end: Option<u32>,
        pcm_bytes: &'a [u8],
    ) -> Self {
        let loop_len = match (loop_start, loop_end) {
            (Some(ls), Some(le)) if le > ls => le - ls,
            _ => 0,
        };
        Self {
            pcm_bytes,
            sample_rate_hz,
            root_key,
            loop_len,
            loop_start,
            loop_end,
        }
    }

    /// Number of unsigned 8-bit PCM samples.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.pcm_bytes.len()
    }

    /// Returns `true` if the sample contains no PCM bytes.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.pcm_bytes.is_empty()
    }

    /// Effective loop length in samples, or `0` if the sample is one-shot.
    #[inline]
    #[must_use]
    pub const fn loop_len(&self) -> usize {
        self.loop_len as usize
    }

    /// Effective playback end offset in samples (`loop_end` if looped, else `len()`).
    #[inline]
    #[must_use]
    pub const fn effective_end(&self) -> usize {
        match self.loop_end {
            Some(le) if (le as usize) <= self.pcm_bytes.len() && le > 0 => le as usize,
            _ => self.pcm_bytes.len(),
        }
    }

    /// Clone into an owned [`SonixSample`] (requires `alloc` feature).
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn to_owned(&self) -> SonixSample {
        SonixSample {
            pcm_bytes: self.pcm_bytes.to_vec(),
            sample_rate_hz: self.sample_rate_hz,
            root_key: self.root_key,
            loop_len: self.loop_len,
            loop_start: self.loop_start,
            loop_end: self.loop_end,
        }
    }

    /// Validate that a loop range fits inside the PCM data.
    pub fn validate_loop(&self) -> Result<()> {
        if let (Some(ls), Some(le)) = (self.loop_start, self.loop_end) {
            if ls > le || (le as usize) > self.pcm_bytes.len() {
                return Err(Error::InvalidLoopRange {
                    loop_start: ls,
                    loop_end: le,
                    pcm_length: self.pcm_bytes.len(),
                });
            }
        }
        Ok(())
    }
}

/// Owned Sonix unsigned 8-bit PCM sample (requires `alloc` feature).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SonixSample {
    /// Raw unsigned 8-bit linear PCM bytes.
    pub pcm_bytes: Vec<u8>,
    /// Native playback sample rate in Hz.
    pub sample_rate_hz: u32,
    /// MIDI root key (`0..=127`).
    pub root_key: u8,
    /// Loop length in samples, or `0` for one-shot samples.
    pub loop_len: u32,
    /// Optional loop start offset in samples.
    pub loop_start: Option<u32>,
    /// Optional loop end offset in samples.
    pub loop_end: Option<u32>,
}

#[cfg(feature = "alloc")]
impl SonixSample {
    /// Borrow this owned sample as a [`SonixSampleView`].
    #[inline]
    #[must_use]
    pub fn as_view(&self) -> SonixSampleView<'_> {
        SonixSampleView {
            pcm_bytes: &self.pcm_bytes,
            sample_rate_hz: self.sample_rate_hz,
            root_key: self.root_key,
            loop_len: self.loop_len,
            loop_start: self.loop_start,
            loop_end: self.loop_end,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_view_loop_math() {
        let pcm = [128u8; 256];
        let view = SonixSampleView::from_pcm_bytes(22_050, 60, Some(16), Some(256), &pcm);
        assert_eq!(view.len(), 256);
        assert_eq!(view.loop_len(), 240);
        assert_eq!(view.effective_end(), 256);
        assert!(view.validate_loop().is_ok());
        let desc = SonixSampleDesc::new_looped(0, 240, 0);
        assert!(desc.is_looped());
    }
}
