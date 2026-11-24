//! Fixed-point phase stepping, volume scaling, and GBA Timer0 rate utilities for
//! the Cosmigo Sonix Sound System.

/// GBA master CPU clock frequency in Hz (`2^24 = 16,777,216 Hz`).
pub const GBA_CLOCK_HZ: u32 = 16_777_216;

/// Number of PCM samples mixed and clamped per Sonix render block (`128`).
pub const SONIX_BLOCK_SIZE: usize = 128;

/// Number of 32-bit accumulator words per Sonix block (`64`, two `i16` lanes each).
pub const SONIX_BLOCK_WORDS: usize = 64;

/// Maximum number of Sonix channels processed per render block.
pub const SONIX_MAX_CHANNELS: usize = 24;

/// Bias added to the 20-bit phase integer to form the ARM phase high word (`0xFFF0_0000`).
pub const SONIX_PHASE_BIAS: u32 = 0xFFF0_0000;

/// Packed-sample mask applied on the first-channel store path (`0xFFFF_1FFF`).
pub const SONIX_STORE_MASK: u32 = 0xFFFF_1FFF;

/// Packed-sample mask applied on the accumulate path (`0x0007_0000`, cleared via `bic`).
pub const SONIX_ACCUM_MASK: u32 = 0x0007_0000;

/// Compute the DirectSound sample rate in Hz from a GBA Timer0 reload value (`TM0CNT_L`).
#[inline]
#[must_use]
pub const fn rate_from_timer_reload(reload: u16) -> u32 {
    let ticks = 65_536u32 - (reload as u32);
    (GBA_CLOCK_HZ + (ticks >> 1)) / ticks
}

/// Convert a desired playback frequency in Hz to a Sonix 32-bit step word for the
/// given engine step-shift byte and hardware mix rate.
///
/// The ARM mixer expands the step into a 64-bit phase increment with
/// `step << (step_shift + 6)`; at 1:1 pitch the phase integer advances by one
/// sample per output sample.
#[inline]
#[must_use]
pub fn freq_to_step(freq_hz: f64, mix_rate_hz: f64, step_shift: u8) -> u32 {
    if !freq_hz.is_finite() || !mix_rate_hz.is_finite() || freq_hz <= 0.0 || mix_rate_hz <= 0.0 {
        return 0;
    }
    // Phase integer is 20 bits of a 64-bit accumulator whose low word carries
    // 12 stored fraction bits; a 1.0 advance is step = 2^(26 - step_shift - 6 + 6)
    // in the stored step word: step_word = ratio * 2^(32 - (step_shift + 6)) * 2^6.
    // Directly: 64-bit increment = ratio * 2^32, and increment = step << (shift+6),
    // so step = ratio * 2^(26 - shift).
    let shift = step_shift as i32;
    let exp = 26 - shift;
    let scaled = if exp >= 0 {
        (freq_hz / mix_rate_hz) * ((1u64 << exp) as f64)
    } else {
        (freq_hz / mix_rate_hz) / ((1u64 << (-exp)) as f64)
    };
    if scaled >= (u32::MAX as f64) {
        u32::MAX
    } else if scaled <= 0.0 {
        0
    } else {
        (scaled + 0.5) as u32
    }
}

/// ARM register-shift semantics for `lsl` by a register amount
/// (amount is taken modulo 256; amounts `>= 32` yield `0`).
#[inline]
#[must_use]
pub const fn arm_lsl(value: u32, shift: u32) -> u32 {
    let shift = shift & 0xFF;
    if shift >= 32 {
        0
    } else {
        value << shift
    }
}

/// ARM register-shift semantics for `lsr` by a register amount
/// (amount is taken modulo 256; amounts `>= 32` yield `0`).
#[inline]
#[must_use]
pub const fn arm_lsr(value: u32, shift: u32) -> u32 {
    let shift = shift & 0xFF;
    if shift >= 32 {
        0
    } else {
        value >> shift
    }
}

/// Compute the exact ARM effective channel volume:
/// `((vol_a * vol_b) * (master_sel * master_c)) >> 21` with wrapping `mul`s.
///
/// `master_sel` is `master_a` when the channel select byte is `>= 0x80`, else `master_b`.
#[inline]
#[must_use]
pub const fn effective_volume(vol_a: u8, vol_b: u8, master_sel: u8, master_c: u8) -> u32 {
    let stage = (vol_a as u32).wrapping_mul(vol_b as u32);
    let masters = (master_sel as u32).wrapping_mul(master_c as u32);
    stage.wrapping_mul(masters) >> 21
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_and_volume_helpers() {
        assert_eq!(rate_from_timer_reload(0xFB1A), 13_379);
        // ((64 * 200) * (100 * 255)) >> 21 = 326_400_000 >> 21 = 155
        assert_eq!(effective_volume(64, 200, 100, 255), 155);
        assert_eq!(effective_volume(0, 200, 100, 255), 0);
        assert_eq!(arm_lsl(1, 32), 0);
        assert_eq!(arm_lsr(0x8000_0000, 31), 1);
        assert_eq!(freq_to_step(22_050.0, 22_050.0, 10), 65_536);
    }
}
