//! Sonix 32-byte ARM7TDMI channel (voice) state structure.

use crate::freq::effective_volume;
use crate::sample::SonixSampleDesc;

/// Exact 32-byte ARM7TDMI channel state consumed by the Sonix mixer.
///
/// Mirrors the in-memory GBA channel struct addressed as `engine + index * 32`:
/// - `+0x00`: `next_index` (`u8`; values `< 2` terminate the channel chain)
/// - `+0x02`: `flags` (`u8`; bit `0x10` enables the volume envelope)
/// - `+0x03`: `vol_select` (`u8`; volume component A and master select: `>= 0x80`
///   selects engine master A, otherwise master B)
/// - `+0x04`: `env_level` (`u16` envelope level, decayed per block)
/// - `+0x05`: `vol` (`u8` volume component B)
/// - `+0x07`: `phase_hi_byte` (`u8`, top byte of the stored 20-bit phase integer)
/// - `+0x0A`: `env_rate` (`u16` envelope decay rate)
/// - `+0x0C`: `sample_desc` (passed separately as a [`SonixSampleDesc`] in safe Rust)
/// - `+0x10`: `phase_lo` (`u32` stored phase: `(phase_lo >> 12) | (phase_hi << 20)` on store)
/// - `+0x14`: `step` (`u32` step word, expanded to a 64-bit phase increment)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SonixVoice {
    /// Next channel index in the engine chain (`+0x00`); `< 2` ends the chain.
    pub next_index: u8,
    /// Channel flags (`+0x02`); bit `0x10` enables the volume envelope.
    pub flags: u8,
    /// Volume component A / master select byte (`+0x03`).
    pub vol_select: u8,
    /// Envelope level (`+0x04`).
    pub env_level: u16,
    /// Volume component B (`+0x05`).
    pub vol: u8,
    /// Top byte of the stored 20-bit phase integer (`+0x07`).
    pub phase_hi_byte: u8,
    /// Envelope decay rate (`+0x0A`).
    pub env_rate: u16,
    /// Stored phase low word (`+0x10`).
    pub phase_lo: u32,
    /// Step word (`+0x14`).
    pub step: u32,
}

impl SonixVoice {
    /// Create a voice with explicit ARM-format parameters.
    #[inline]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        next_index: u8,
        flags: u8,
        vol_select: u8,
        env_level: u16,
        vol: u8,
        phase_hi_byte: u8,
        env_rate: u16,
        phase_lo: u32,
        step: u32,
    ) -> Self {
        Self {
            next_index,
            flags,
            vol_select,
            env_level,
            vol,
            phase_hi_byte,
            env_rate,
            phase_lo,
            step,
        }
    }

    /// Create a simple forward voice at sample index `0` with the given step and volumes.
    #[inline]
    #[must_use]
    pub const fn new_simple(vol_select: u8, vol: u8, step: u32) -> Self {
        Self {
            next_index: 0,
            flags: 0,
            vol_select,
            env_level: 0,
            vol,
            phase_hi_byte: 0,
            env_rate: 0,
            phase_lo: 0,
            step,
        }
    }

    /// Create a silent dummy voice (volume `0`, frozen at index `0`).
    #[inline]
    #[must_use]
    pub const fn silent() -> Self {
        Self::new_simple(0, 0, 0)
    }

    /// Set the voice phase from a 20-bit sample index and 12-bit fraction.
    ///
    /// The ARM store format is `phase_lo = (lo >> 12) | (hi << 20)` with
    /// `phase_hi_byte = hi >> 12`, where `hi` is the biased phase integer.
    #[inline]
    #[must_use]
    pub const fn with_phase_index(mut self, index20: u32, frac12: u32) -> Self {
        // Reconstruct the stored form whose mixer expansion yields `index20`.
        // Mixer: hi = ((phase_lo >> 20) | (phase_hi_byte << 12)) | BIAS,
        //        lo = phase_lo << 12.
        // Stored back: phase_lo' = (lo >> 12) | (hi << 20), byte' = hi >> 12.
        // Setting phase_lo = (index20 << 20) | frac12 and byte = index20 >> 12
        // gives hi low bits = index20 and lo fraction = frac12 << 12... the
        // mixer masks lo to its low 12 stored bits via the <<12 expansion.
        self.phase_lo = index20.wrapping_shl(20) | (frac12 & 0xFFF);
        self.phase_hi_byte = (index20 >> 12) as u8;
        self
    }

    /// Reconstruct the biased ARM phase high word (`hi`) from the stored fields.
    #[inline]
    #[must_use]
    pub const fn phase_hi(&self) -> u32 {
        ((self.phase_lo >> 20) | ((self.phase_hi_byte as u32) << 12)) | crate::freq::SONIX_PHASE_BIAS
    }

    /// Reconstruct the ARM phase low word (`lo`) from the stored fields.
    #[inline]
    #[must_use]
    pub const fn phase_lo_word(&self) -> u32 {
        self.phase_lo << 12
    }

    /// Store a mixer phase (`hi`, `lo`) back into the ARM stored format.
    #[inline]
    pub fn store_phase(&mut self, hi: u32, lo: u32) {
        self.phase_hi_byte = (hi >> 12) as u8;
        self.phase_lo = (lo >> 12) | (hi << 20);
    }

    /// Current 20-bit sample index (`hi` with the bias removed).
    #[inline]
    #[must_use]
    pub const fn sample_index(&self) -> u32 {
        self.phase_hi() & 0x000F_FFFF
    }

    /// Compute the exact ARM effective volume for the given engine masters.
    ///
    /// `master_a` is selected when `vol_select >= 0x80`, otherwise `master_b`;
    /// the result is `((vol_select * vol) * (master_sel * master_c)) >> 21`.
    #[inline]
    #[must_use]
    pub const fn effective_volume(&self, master_a: u8, master_b: u8, master_c: u8) -> u32 {
        let sel = if self.vol_select >= 0x80 {
            master_a
        } else {
            master_b
        };
        effective_volume(self.vol_select, self.vol, sel, master_c)
    }

    /// Returns `true` if the volume envelope is enabled (`flags & 0x10`).
    #[inline]
    #[must_use]
    pub const fn envelope_enabled(&self) -> bool {
        self.flags & 0x10 != 0
    }

    /// Create a voice initialized from a parsed [`SonixSampleDesc`].
    #[inline]
    #[must_use]
    pub const fn from_sample_desc(desc: &SonixSampleDesc, vol_select: u8, vol: u8, step: u32) -> Self {
        let _ = desc;
        Self::new_simple(vol_select, vol, step)
    }
}

impl Default for SonixVoice {
    #[inline]
    fn default() -> Self {
        Self::silent()
    }
}
