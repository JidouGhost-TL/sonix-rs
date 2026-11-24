//! Dual Synthesis Engines for the Cosmigo Sonix Sound System:
//!
//! - [`AccuracySynth`]: 100% bit-exact emulation of all three Sonix ARM7TDMI mixer
//!   releases (`V109c`, `V1096Gsm`, `V116Gsm`, which share identical mixer math),
//!   verified strictly byte-for-byte (`assert_eq!`, 0 tolerance) against headless
//!   `libmgba.so.0.10` ARM7TDMI execution.
//! - [`StudioSynth`]: High-fidelity 48,000 Hz `f32` software synthesizer featuring a
//!   compile-time 256-phase × 6-tap Lanczos-3 polyphase windowed-sinc LUT (or general
//!   Lanczos-`a` fallback) and true equal-power stereo panning.

use crate::error::{Error, Result};
use crate::freq::{arm_lsl, arm_lsr, SONIX_BLOCK_SIZE, SONIX_BLOCK_WORDS, SONIX_PHASE_BIAS};
use crate::version::SonixVersion;
use crate::sample::SonixSampleView;
use crate::voice::SonixVoice;

// ============================================================================
// Pure no_std math & compile-time Lanczos-3 polyphase LUT
// ============================================================================

const LANCZOS_PHASES: usize = 256;
const LANCZOS3_TAPS: usize = 6;

/// 32-byte aligned Lanczos-3 polyphase filter bank (`256` phases × 8 `f32` lanes = 8 KiB,
/// padded with two zero lanes for shift-by-5 indexing and L1-friendly alignment).
#[repr(C, align(32))]
struct Lanczos3Table([[f32; 8]; LANCZOS_PHASES]);

/// Pure `const fn` sine approximation on `[-pi, pi]` (range-reduced, < 1e-13 max error).
#[inline]
const fn sin_const(mut x: f64) -> f64 {
    const PI: f64 = core::f64::consts::PI;
    const TWO_PI: f64 = 2.0 * PI;
    const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;

    if x > PI || x < -PI {
        let k = ((x + PI) / TWO_PI) as i64;
        x -= (k as f64) * TWO_PI;
        if x > PI {
            x -= TWO_PI;
        } else if x < -PI {
            x += TWO_PI;
        }
    }
    if x > HALF_PI {
        x = PI - x;
    } else if x < -HALF_PI {
        x = -PI - x;
    }

    let x2 = x * x;
    let mut term = x;
    let mut sum = x;
    let mut n = 1u32;
    while n <= 9 {
        let d = (2 * n) * (2 * n + 1);
        term = -term * x2 / (d as f64);
        sum += term;
        n += 1;
    }
    sum
}

#[inline]
const fn cos_const(x: f64) -> f64 {
    sin_const(x + core::f64::consts::FRAC_PI_2)
}

#[inline]
const fn lanczos_kernel_const(x: f64, a: f64) -> f64 {
    let ax = if x < 0.0 { -x } else { x };
    if ax < 1e-12 {
        return 1.0;
    }
    if ax >= a {
        return 0.0;
    }
    let px = core::f64::consts::PI * x;
    let px_a = px / a;
    (sin_const(px) / px) * (sin_const(px_a) / px_a)
}

/// Build the normalized 256-entry (`0..256`) × 6-tap (padded to 8) Lanczos-3 polyphase filter
/// table entirely at compile time.
const fn build_lanczos3_lut() -> Lanczos3Table {
    let mut table = [[0.0f32; 8]; LANCZOS_PHASES];
    let mut p = 0usize;
    while p < LANCZOS_PHASES {
        let frac = (p as f64) / (LANCZOS_PHASES as f64);
        let mut weights = [0.0f64; LANCZOS3_TAPS];
        let mut wsum = 0.0f64;
        let mut t = 0usize;
        while t < LANCZOS3_TAPS {
            let k = (t as isize) - 2; // taps at -2, -1, 0, +1, +2, +3
            let w = lanczos_kernel_const(frac - (k as f64), 3.0);
            weights[t] = w;
            wsum += w;
            t += 1;
        }
        let inv = if wsum > 1e-12 { 1.0 / wsum } else { 1.0 };
        let mut t2 = 0usize;
        while t2 < LANCZOS3_TAPS {
            table[p][t2] = (weights[t2] * inv) as f32;
            t2 += 1;
        }
        p += 1;
    }
    Lanczos3Table(table)
}

static LANCZOS3_LUT: Lanczos3Table = build_lanczos3_lut();

// ============================================================================
// AccuracySynth — 100% Bit-Exact ARM7TDMI Sonix Emulation
// ============================================================================

/// Engine-level mixer context (the ARM `ip` struct fields used by the mixer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SonixContext {
    /// Engine master volume A (`+0x05`), selected when `vol_select >= 0x80`.
    pub master_a: u8,
    /// Engine master volume B (`+0x06`), selected when `vol_select < 0x80`.
    pub master_b: u8,
    /// Engine master volume C (`+0x07`), always multiplied into the volume.
    pub master_c: u8,
    /// Engine step-shift byte (`+0x0C`); the 64-bit step is `step << (step_shift + 6)`.
    pub step_shift: u8,
}

impl Default for SonixContext {
    #[inline]
    fn default() -> Self {
        Self {
            master_a: 255,
            master_b: 255,
            master_c: 255,
            step_shift: 10,
        }
    }
}

/// Byte-exact ARM7TDMI software mixer emulation for all three Sonix releases.
///
/// The three releases share identical mixer math; they differ only in register
/// allocation, engine header size, and GSM codec presence, none of which affect
/// the mixed output bytes.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct AccuracySynth;

impl AccuracySynth {
    /// Advance the 64-bit ARM phase by the voice step, applying loop wrap.
    ///
    /// Mirrors `adds lo, lo, step_lo; adcs hi, hi, step_hi; bhs wrap_subtract_loop`:
    /// on carry out of the high word, `loop_len` is subtracted until borrow.
    #[inline]
    fn advance_phase(hi: u32, lo: u32, step_hi: u32, step_lo: u32, loop_len: u32) -> (u32, u32) {
        let (new_lo, carry) = lo.overflowing_add(step_lo);
        let hi_sum = (hi as u64) + (step_hi as u64) + (carry as u64);
        let new_hi = hi_sum as u32;
        if hi_sum > 0xFFFF_FFFF && loop_len != 0 {
            let mut h = new_hi;
            loop {
                let (nh, borrow) = h.overflowing_sub(loop_len);
                h = nh;
                if borrow {
                    break;
                }
            }
            (h, new_lo)
        } else {
            (new_hi, new_lo)
        }
    }

    /// Mix one Sonix channel into the 128-sample accumulator block.
    ///
    /// - `acc`: accumulator of 64 `u32` words, each holding two `i16` lanes.
    /// - `pcm`: unsigned 8-bit sample data; fetch index is the 20-bit phase integer.
    /// - `loop_len`: sample loop length in phase-integer units (`0` = one-shot).
    /// - `voice`: channel state; phase and envelope level are updated in place.
    /// - `ctx`: engine masters and step-shift byte.
    /// - `accumulate`: `false` for the first channel in a block (ARM store path,
    ///   `(packed * vol >> 3) & 0xFFFF_1FFF`), `true` for later channels (ARM
    ///   accumulate path, `acc += (packed * vol & !0x7_0000) >> 3`).
    /// - `count`: samples to mix in this block (`1..=128`).
    ///
    /// Returns the channel's effective volume contribution to the block total.
    #[allow(clippy::too_many_arguments)]
    pub fn mix_channel(
        version: SonixVersion,
        acc: &mut [u32; SONIX_BLOCK_WORDS],
        pcm: &[u8],
        loop_len: u32,
        voice: &mut SonixVoice,
        ctx: &SonixContext,
        accumulate: bool,
        count: usize,
    ) -> u32 {
        let count = count.min(SONIX_BLOCK_SIZE);
        if count == 0 {
            return 0;
        }
        let eff_vol = voice.effective_volume(ctx.master_a, ctx.master_b, ctx.master_c);

        // Volume envelope decay (ARM: level -= (rate * count) >> (8 - step_shift)).
        if voice.envelope_enabled() {
            let shift = 8u32.wrapping_sub(ctx.step_shift as u32);
            let decay = arm_lsr((voice.env_rate as u32).wrapping_mul(count as u32), shift);
            // ARM `subs` + `movmi #0`: a negative 32-bit result clamps to zero.
            voice.env_level = (voice.env_level as u32).saturating_sub(decay) as u16;
        }

        // 64-bit phase and step expansion.
        let mut hi = voice.phase_hi();
        let mut lo = voice.phase_lo_word();
        let shift_amt = (ctx.step_shift as u32) + 6;
        let step_lo = arm_lsl(voice.step, shift_amt);
        let step_hi = arm_lsr(voice.step, 32u32.wrapping_sub(shift_amt) & 0xFF);
        // ARM `lsr` by (32 - shift_amt): when shift_amt > 32 the amount wraps;
        // the helper already implements register-shift semantics for the
        // wrapped amount computed above. For shift_amt == 0 (unreachable for
        // byte step_shift, kept for completeness) the amount is 32 -> 0.

        let fetch = |hi: u32| -> u32 {
            let idx = hi.wrapping_sub(SONIX_PHASE_BIAS) as usize;
            if idx < pcm.len() {
                pcm[idx] as u32
            } else {
                0
            }
        };

        let mut word_idx = 0usize;
        let mut remaining = count as i32;
        while remaining > 0 {
            // Fetch 4 samples with per-sample phase advance and loop wrap.
            let s0 = fetch(hi);
            let (h, l) = Self::advance_phase(hi, lo, step_hi, step_lo, loop_len);
            hi = h;
            lo = l;
            let s1 = fetch(hi);
            let (h, l) = Self::advance_phase(hi, lo, step_hi, step_lo, loop_len);
            hi = h;
            lo = l;
            let s2 = fetch(hi);
            let (h, l) = Self::advance_phase(hi, lo, step_hi, step_lo, loop_len);
            hi = h;
            lo = l;
            let s3 = fetch(hi);
            let (h, l) = Self::advance_phase(hi, lo, step_hi, step_lo, loop_len);
            hi = h;
            lo = l;

            let packed0 = s0 | (s1 << 16);
            let packed1 = s2 | (s3 << 16);
            let shift = version.mix_shift();
            if accumulate {
                let w0 = (packed0.wrapping_mul(eff_vol) & !version.accum_mask()) >> shift;
                let w1 = (packed1.wrapping_mul(eff_vol) & !version.accum_mask()) >> shift;
                if word_idx < SONIX_BLOCK_WORDS {
                    acc[word_idx] = acc[word_idx].wrapping_add(w0);
                }
                if word_idx + 1 < SONIX_BLOCK_WORDS {
                    acc[word_idx + 1] = acc[word_idx + 1].wrapping_add(w1);
                }
            } else {
                let w0 = (packed0.wrapping_mul(eff_vol) >> shift) & version.store_mask();
                let w1 = (packed1.wrapping_mul(eff_vol) >> shift) & version.store_mask();
                if word_idx < SONIX_BLOCK_WORDS {
                    acc[word_idx] = w0;
                }
                if word_idx + 1 < SONIX_BLOCK_WORDS {
                    acc[word_idx + 1] = w1;
                }
            }
            word_idx += 2;
            remaining -= 4;
        }

        voice.store_phase(hi, lo);
        eff_vol
    }

    /// Clamp the 128-sample accumulator block into signed 8-bit DMA output.
    ///
    /// Mirrors the ARM clamp stage exactly, including its two different
    /// scaling paths: even samples use the `((acc << 16) - (total << 20))`
    /// scaled-clamp path, odd samples use the `(acc >> 20) - total` path.
    /// Both compute `clamp((acc >> 4) - total_volume, -128, 127)` for in-range
    /// accumulator values and saturate identically outside it.
    pub fn clamp_block(
        version: SonixVersion,
        acc: &[u32; SONIX_BLOCK_WORDS],
        total_volume: u16,
        out: &mut [i8; SONIX_BLOCK_SIZE],
    ) {
        let shift = version.clamp_shift();
        let neg_total = (total_volume as u32).wrapping_neg();
        let neg_total_scaled = neg_total << shift;
        for (pair, chunk) in acc.iter().enumerate() {
            let w = *chunk;
            // Low i16 lane (even sample): scaled path.
            let low_tmp = neg_total_scaled.wrapping_add(w << 16) as i32;
            let low_clamped = low_tmp.clamp(-128 << shift, 127 << shift);
            let low_byte = (((low_clamped as u32) >> shift) & 0xFF) as u8 as i8;
            // High i16 lane (odd sample): direct path.
            let high_val = neg_total.wrapping_add(w >> shift) as i32;
            let high_byte = high_val.clamp(-128, 127) as i8;
            out[pair * 2] = low_byte;
            out[pair * 2 + 1] = high_byte;
        }
    }

    /// Render one 128-sample block through a channel chain.
    ///
    /// `voices` is the engine channel table indexed by ARM channel index;
    /// `samples[i]` is the `(pcm, loop_len)` pair for `voices[i]`. The chain
    /// starts at `first_index` and follows each voice's `next_index` until the
    /// count is exhausted or a voice reports `next_index < 2`. The first mixed
    /// channel uses the store path; later channels accumulate.
    ///
    /// Returns the block total volume (sum of effective channel volumes).
    #[allow(clippy::too_many_arguments)]
    pub fn render_block(
        version: SonixVersion,
        voices: &mut [SonixVoice],
        samples: &[(&[u8], u32)],
        first_index: usize,
        channel_count: usize,
        ctx: &SonixContext,
        acc: &mut [u32; SONIX_BLOCK_WORDS],
        out: &mut [i8; SONIX_BLOCK_SIZE],
        count: usize,
    ) -> Result<u16> {
        if first_index >= voices.len() || first_index >= samples.len() {
            return Err(Error::InvalidChannel(first_index));
        }
        let mut total: u32 = 0;
        let mut index = first_index;
        let mut remaining = channel_count;
        let mut first = true;
        loop {
            if index >= voices.len() || index >= samples.len() {
                return Err(Error::InvalidChannel(index));
            }
            let (pcm, loop_len) = samples[index];
            // Split borrows: copy the voice out, mix, write back.
            let mut voice = voices[index];
            let eff = Self::mix_channel(version, acc, pcm, loop_len, &mut voice, ctx, !first, count);
            voices[index] = voice;
            total = total.wrapping_add(eff);
            first = false;
            remaining = remaining.saturating_sub(1);
            if remaining == 0 {
                break;
            }
            let next = voices[index].next_index as usize;
            if next < 2 {
                break;
            }
            index = next;
        }
        let total16 = total as u16;
        Self::clamp_block(version, acc, total16, out);
        Ok(total16)
    }
}

// ============================================================================
// StudioSynth — 48 kHz f32 Lanczos-3 Polyphase LUT Renderer
// ============================================================================

/// High-fidelity studio software synthesizer for Sonix samples.
///
/// Renders unsigned 8-bit PCM samples at 48,000 Hz (configurable) in 32-bit
/// floating-point using a compile-time 256-phase × 6-tap Lanczos-3 windowed-sinc
/// polyphase LUT (or general Lanczos-`a` fallback when `sinc_lobes != 3`) with
/// true equal-power stereo panning. Sample bytes are centred as
/// `(byte - 128) / 128`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StudioSynth {
    /// Output sample rate in Hz (default `48_000`).
    pub sample_rate: u32,
    /// Number of Lanczos sinc lobes (`3` uses the compile-time polyphase LUT fast path).
    pub sinc_lobes: usize,
}

impl Default for StudioSynth {
    fn default() -> Self {
        Self::new(Self::SAMPLE_RATE)
    }
}

impl StudioSynth {
    /// Standard studio output sample rate (`48_000` Hz).
    pub const SAMPLE_RATE: u32 = 48_000;

    /// Create a new `StudioSynth` at the specified output sample rate (typically `48_000` Hz).
    #[inline]
    #[must_use]
    pub const fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            sinc_lobes: 3,
        }
    }

    /// Render a Sonix sample into stereo `f32` buffers (accumulating) using band-limited
    /// Lanczos windowed-sinc interpolation and true equal-power stereo panning.
    ///
    /// - `freq_hz`: desired playback sample rate in Hz.
    /// - `volume`: linear gain in `0.0..=1.0`.
    /// - `pan`: stereo pan in `-1.0..=1.0` (`-1.0` = hard left, `0.0` = center, `+1.0` = hard right).
    #[allow(clippy::too_many_arguments)]
    pub fn render_voice_f32(
        &self,
        sample: &SonixSampleView<'_>,
        pos_f64: &mut f64,
        freq_hz: f64,
        volume: f32,
        pan: f32,
        out_left: &mut [f32],
        out_right: &mut [f32],
    ) {
        let frames = out_left.len().min(out_right.len());
        if frames == 0
            || sample.is_empty()
            || self.sample_rate == 0
            || !freq_hz.is_finite()
            || freq_hz <= 0.0
            || !pos_f64.is_finite()
        {
            return;
        }
        if *pos_f64 < 0.0 {
            *pos_f64 = 0.0;
        }

        let step = freq_hz / (self.sample_rate as f64);
        let gain = volume.clamp(0.0, 1.0) as f64;
        // Equal-power stereo pan law: pan in [-1.0, +1.0] -> theta in [0, pi/2]
        let pan_norm = ((pan.clamp(-1.0, 1.0) as f64) + 1.0) * 0.5;
        let angle = pan_norm * core::f64::consts::FRAC_PI_2;
        let gain_l = (gain * cos_const(angle)) as f32;
        let gain_r = (gain * sin_const(angle)) as f32;

        let pcm_end = sample.effective_end();
        if pcm_end == 0 {
            return;
        }
        let pcm = &sample.pcm_bytes[..pcm_end];
        let pcm_len = pcm.len();
        let loop_len = sample.loop_len();
        let looped = loop_len > 0;
        let loop_start = sample.loop_start.unwrap_or(0) as usize;
        let loop_start_f = loop_start as f64;
        let loop_len_f = loop_len as f64;
        let pcm_len_f = pcm_len as f64;

        let out_l = &mut out_left[..frames];
        let out_r = &mut out_right[..frames];

        let sample_at = |pcm: &[u8], idx: isize| -> f64 {
            if idx < 0 || idx as usize >= pcm.len() {
                0.0
            } else {
                ((pcm[idx as usize] as f64) - 128.0) / 128.0
            }
        };

        if self.sinc_lobes == 3 {
            let lut = &LANCZOS3_LUT.0;
            let mut ph = *pos_f64;
            for idx in 0..frames {
                if ph >= pcm_len_f {
                    if looped {
                        ph = loop_start_f + (ph - loop_start_f) % loop_len_f;
                    } else {
                        break;
                    }
                }
                // ph is always >= 0 here, so truncation is floor.
                let base = (ph as i64) as f64;
                let frac = ph - base;
                let base_i = base as isize;
                let phase = (frac * (LANCZOS_PHASES as f64)) as usize;
                let weights = &lut[phase.min(LANCZOS_PHASES - 1)];
                let mut acc = 0.0f64;
                if base_i >= 2 && (base_i as usize) + 4 < pcm_len {
                    // Interior fast path: all six taps are in range.
                    for (t, &w) in weights.iter().take(LANCZOS3_TAPS).enumerate() {
                        let k = (t as isize) - 2;
                        let s = ((pcm[(base_i + k) as usize] as f64) - 128.0) / 128.0;
                        acc += (w as f64) * s;
                    }
                } else {
                    for (t, &w) in weights.iter().take(LANCZOS3_TAPS).enumerate() {
                        let k = (t as isize) - 2;
                        acc += (w as f64) * sample_at(pcm, base_i + k);
                    }
                }
                out_l[idx] += (acc as f32) * gain_l;
                out_r[idx] += (acc as f32) * gain_r;
                ph += step;
            }
            *pos_f64 = ph;
        } else {
            let a = self.sinc_lobes as f64;
            let taps = self.sinc_lobes * 2;
            let mut ph = *pos_f64;
            for idx in 0..frames {
                if ph >= pcm_len_f {
                    if looped {
                        ph = loop_start_f + (ph - loop_start_f) % loop_len_f;
                    } else {
                        break;
                    }
                }
                // ph is always >= 0 here, so truncation is floor.
                let base = (ph as i64) as f64;
                let frac = ph - base;
                let base_i = base as isize;
                let mut acc = 0.0f64;
                let mut wsum = 0.0f64;
                for t in 0..taps {
                    let k = (t as isize) - (self.sinc_lobes as isize) + 1;
                    let w = lanczos_kernel_const(frac - (k as f64), a);
                    acc += w * sample_at(pcm, base_i + k);
                    wsum += w;
                }
                if wsum.abs() > 1e-12 {
                    acc /= wsum;
                }
                out_l[idx] += (acc as f32) * gain_l;
                out_r[idx] += (acc as f32) * gain_r;
                ph += step;
            }
            *pos_f64 = ph;
        }
    }
}
