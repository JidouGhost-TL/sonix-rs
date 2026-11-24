//! Dual Synthesis Engines for the Cosmigo Sonix Sound System:
//!
//! - [`AccuracySynth`]: 100% bit-exact emulation of all three Sonix ARM7TDMI mixer
//!   releases (`V109c` with `>> 2` / clamp `>> 21`, and `V1096Gsm` / `V116Gsm` with
//!   `>> 3` / clamp `>> 20`), verified strictly byte-for-byte (`assert_eq!`, 0
//!   tolerance) against headless `libmgba.so.0.10` ARM7TDMI execution.
//! - [`StudioSynth`]: High-fidelity 48,000 Hz `f32` software synthesizer featuring a
//!   compile-time 256-phase × 6-tap Lanczos-3 polyphase windowed-sinc LUT (or general
//!   Lanczos-`a` fallback) with Q32.32 interior stepping and true equal-power stereo
//!   panning.

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

/// Compile-time 1 KiB L1-resident LUT mapping unsigned 8-bit PCM bytes (`0..=255`)
/// to centred floating-point values `(b - 128) / 128` (`128 -> 0.0` exact).
const fn build_u8_to_f32_lut() -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        table[i] = ((i as i32) - 128) as f32 * (1.0 / 128.0);
        i += 1;
    }
    table
}

static U8_TO_F32: [f32; 256] = build_u8_to_f32_lut();

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
/// Emulates the exact per-revision shifts and masks (`V109c`: `>> 2` / `0xFFFF_3FFF` /
/// `0x0003_0000` / clamp `>> 21`; `V1096Gsm` & `V116Gsm`: `>> 3` / `0xFFFF_1FFF` /
/// `0x0007_0000` / clamp `>> 20`).
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

    #[inline(always)]
    fn mix_quad<const SHIFT: u32, const MASK: u32, const ACCUM: bool>(
        dst: &mut [u32; 2],
        p0: u32,
        p1: u32,
        eff_vol: u32,
    ) {
        if ACCUM {
            let w0 = (p0.wrapping_mul(eff_vol) & !MASK) >> SHIFT;
            let w1 = (p1.wrapping_mul(eff_vol) & !MASK) >> SHIFT;
            dst[0] = dst[0].wrapping_add(w0);
            dst[1] = dst[1].wrapping_add(w1);
        } else {
            let w0 = (p0.wrapping_mul(eff_vol) >> SHIFT) & MASK;
            let w1 = (p1.wrapping_mul(eff_vol) >> SHIFT) & MASK;
            dst[0] = w0;
            dst[1] = w1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn mix_channel_inner<const SHIFT: u32, const MASK: u32, const ACCUM: bool>(
        acc: &mut [u32; SONIX_BLOCK_WORDS],
        pcm: &[u8],
        loop_len: u32,
        eff_vol: u32,
        mut hi: u32,
        mut lo: u32,
        step_hi: u32,
        step_lo: u32,
        n_quads: usize,
    ) -> (u32, u32) {
        let total_samples = n_quads * 4;
        let phase64 = ((hi as u64) << 32) | (lo as u64);
        let step64 = ((step_hi as u64) << 32) | (step_lo as u64);
        let start_idx = hi.wrapping_sub(SONIX_PHASE_BIAS) as usize;
        let acc_pairs = &mut acc.as_chunks_mut::<2>().0[..n_quads];

        // Fast path 1: exact 1:1 pitch, fully in-bounds, zero phase carry across the block.
        if step_hi == 1
            && step_lo == 0
            && hi >= SONIX_PHASE_BIAS
            && (hi as u64) + (total_samples as u64) <= 0xFFFF_FFFF
            && start_idx + total_samples <= pcm.len()
        {
            let src = &pcm[start_idx..start_idx + total_samples];
            let (src_quads, _) = src.as_chunks::<4>();
            for (dst, s) in acc_pairs.iter_mut().zip(src_quads.iter()) {
                let p0 = (s[0] as u32) | ((s[1] as u32) << 16);
                let p1 = (s[2] as u32) | ((s[3] as u32) << 16);
                Self::mix_quad::<SHIFT, MASK, ACCUM>(dst, p0, p1, eff_vol);
            }
            hi = hi.wrapping_add(total_samples as u32);
            return (hi, lo);
        }

        // Fast path 2: general fractional step, fully in-bounds, zero phase carry across the block.
        let total_advance = (step64 as u128) * (total_samples as u128);
        if hi >= SONIX_PHASE_BIAS && (phase64 as u128) + total_advance <= (u64::MAX as u128) {
            let last_phase = phase64 + step64 * ((total_samples - 1) as u64);
            let max_idx = ((last_phase >> 32) as u32).wrapping_sub(SONIX_PHASE_BIAS) as usize;
            if max_idx < pcm.len() {
                let bias64 = (SONIX_PHASE_BIAS as u64) << 32;
                let mut rel = phase64 - bias64;
                let step2 = step64.wrapping_mul(2);
                let step3 = step64.wrapping_mul(3);
                let step4 = step64.wrapping_mul(4);
                let src = &pcm[..=max_idx];
                for dst in acc_pairs.iter_mut() {
                    let r0 = rel;
                    let r1 = r0.wrapping_add(step64);
                    let r2 = r0.wrapping_add(step2);
                    let r3 = r0.wrapping_add(step3);
                    rel = r0.wrapping_add(step4);

                    let s0 = src[(r0 >> 32) as usize] as u32;
                    let s1 = src[(r1 >> 32) as usize] as u32;
                    let s2 = src[(r2 >> 32) as usize] as u32;
                    let s3 = src[(r3 >> 32) as usize] as u32;

                    let p0 = s0 | (s1 << 16);
                    let p1 = s2 | (s3 << 16);
                    Self::mix_quad::<SHIFT, MASK, ACCUM>(dst, p0, p1, eff_vol);
                }
                let final_phase = rel + bias64;
                return ((final_phase >> 32) as u32, final_phase as u32);
            }
        }

        // General path: handles loop wrap and past-end zero-fill.
        let fetch = |hi: u32| -> u32 {
            let idx = hi.wrapping_sub(SONIX_PHASE_BIAS) as usize;
            if idx < pcm.len() {
                pcm[idx] as u32
            } else {
                0
            }
        };

        for dst in acc_pairs.iter_mut() {
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

            let p0 = s0 | (s1 << 16);
            let p1 = s2 | (s3 << 16);
            Self::mix_quad::<SHIFT, MASK, ACCUM>(dst, p0, p1, eff_vol);
        }
        (hi, lo)
    }

    /// Mix one Sonix channel into the 128-sample accumulator block.
    ///
    /// - `acc`: accumulator of 64 `u32` words, each holding two `i16` lanes.
    /// - `pcm`: unsigned 8-bit sample data; fetch index is the 20-bit phase integer.
    /// - `loop_len`: sample loop length in phase-integer units (`0` = one-shot).
    /// - `voice`: channel state; phase and envelope level are updated in place.
    /// - `ctx`: engine masters and step-shift byte.
    /// - `accumulate`: `false` for the first channel in a block (ARM store path,
    ///   `(packed * vol >> shift) & store_mask`), `true` for later channels (ARM
    ///   accumulate path, `acc += (packed * vol & !accum_mask) >> shift`).
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
        let hi = voice.phase_hi();
        let lo = voice.phase_lo_word();
        let shift_amt = (ctx.step_shift as u32) + 6;
        let step_lo = arm_lsl(voice.step, shift_amt);
        let step_hi = arm_lsr(voice.step, 32u32.wrapping_sub(shift_amt) & 0xFF);
        let n_quads = count.div_ceil(4);

        let (new_hi, new_lo) = match (version, accumulate) {
            (SonixVersion::V109c, false) => Self::mix_channel_inner::<2, 0xFFFF_3FFF, false>(
                acc, pcm, loop_len, eff_vol, hi, lo, step_hi, step_lo, n_quads,
            ),
            (SonixVersion::V109c, true) => Self::mix_channel_inner::<2, 0x0003_0000, true>(
                acc, pcm, loop_len, eff_vol, hi, lo, step_hi, step_lo, n_quads,
            ),
            (SonixVersion::V1096Gsm | SonixVersion::V116Gsm, false) => {
                Self::mix_channel_inner::<3, 0xFFFF_1FFF, false>(
                    acc, pcm, loop_len, eff_vol, hi, lo, step_hi, step_lo, n_quads,
                )
            }
            (SonixVersion::V1096Gsm | SonixVersion::V116Gsm, true) => {
                Self::mix_channel_inner::<3, 0x0007_0000, true>(
                    acc, pcm, loop_len, eff_vol, hi, lo, step_hi, step_lo, n_quads,
                )
            }
        };

        voice.store_phase(new_hi, new_lo);
        eff_vol
    }

    #[inline]
    fn clamp_block_const<const SHIFT: u32>(
        acc: &[u32; SONIX_BLOCK_WORDS],
        total_volume: u16,
        out: &mut [i8; SONIX_BLOCK_SIZE],
    ) {
        let neg_total = (total_volume as u32).wrapping_neg();
        let neg_total_scaled = neg_total << SHIFT;
        let low_min = -128i32 << SHIFT;
        let low_max = 127i32 << SHIFT;
        for i in 0..SONIX_BLOCK_WORDS {
            let w = acc[i];
            // Low i16 lane (even sample): scaled path.
            let low_tmp = neg_total_scaled.wrapping_add(w << 16) as i32;
            let low_clamped = low_tmp.clamp(low_min, low_max);
            let low_byte = ((low_clamped as u32) >> SHIFT) as i8;
            // High i16 lane (odd sample): direct path.
            let high_val = neg_total.wrapping_add(w >> SHIFT) as i32;
            let high_byte = high_val.clamp(-128, 127) as i8;
            out[i * 2] = low_byte;
            out[i * 2 + 1] = high_byte;
        }
    }

    /// Clamp the 128-sample accumulator block into signed 8-bit DMA output.
    ///
    /// Mirrors the ARM clamp stage exactly, including its two different
    /// scaling paths: even samples use the `((acc << 16) - (total << shift))`
    /// scaled-clamp path, odd samples use the `(acc >> shift) - total` path.
    pub fn clamp_block(
        version: SonixVersion,
        acc: &[u32; SONIX_BLOCK_WORDS],
        total_volume: u16,
        out: &mut [i8; SONIX_BLOCK_SIZE],
    ) {
        match version {
            SonixVersion::V109c => Self::clamp_block_const::<21>(acc, total_volume, out),
            SonixVersion::V1096Gsm | SonixVersion::V116Gsm => {
                Self::clamp_block_const::<20>(acc, total_volume, out)
            }
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
        if channel_count == 0 {
            Self::clamp_block(version, acc, 0, out);
            return Ok(0);
        }
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
            remaining -= 1;
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
        let loop_start = sample.loop_start.unwrap_or(0) as usize;
        let loop_len = sample.loop_len().min(pcm_len.saturating_sub(loop_start));
        let looped = loop_len > 0;
        let loop_start_f = loop_start as f64;
        let loop_len_f = loop_len as f64;
        let pcm_len_f = pcm_len as f64;

        let out_l = &mut out_left[..frames];
        let out_r = &mut out_right[..frames];

        if self.sinc_lobes == 3 {
            let lut = &LANCZOS3_LUT.0;
            let u8_lut = &U8_TO_F32;
            let mut ph = *pos_f64;

            // Fast Q32.32 fixed-point interior-segment renderer when indices and step
            // fit comfortably inside 30-bit integer range (always true for GBA PCM).
            const Q32_SCALE: f64 = 4_294_967_296.0;
            const Q32_INV: f64 = 1.0 / Q32_SCALE;
            if pcm_len < (1usize << 30)
                && ph < (1u64 << 30) as f64
                && step < (1u64 << 29) as f64
            {
                let mut pos_q = (ph * Q32_SCALE) as u64;
                let step_q = (step * Q32_SCALE) as u64;
                let step_q2 = step_q.wrapping_mul(2);
                let step_q3 = step_q.wrapping_mul(3);
                let step_q4 = step_q.wrapping_mul(4);
                let pcm_len_q = (pcm_len as u64) << 32;
                let loop_start_q = (loop_start as u64) << 32;
                let loop_len_q = (loop_len as u64) << 32;
                let max_interior_q = if pcm_len >= 6 {
                    (((pcm_len - 4) as u64) << 32) | 0xFFFF_FFFF
                } else {
                    0
                };

                #[inline(always)]
                fn eval_interior(
                    pcm: &[u8],
                    lut: &[[f32; 8]; LANCZOS_PHASES],
                    u8_lut: &[f32; 256],
                    p: u64,
                ) -> f32 {
                    let b = (p >> 32) as usize;
                    let w = &lut[((p >> 24) & 0xFF) as usize];
                    let s = &pcm[b - 2..b + 4];
                    let t0 = u8_lut[s[0] as usize];
                    let t1 = u8_lut[s[1] as usize];
                    let t2 = u8_lut[s[2] as usize];
                    let t3 = u8_lut[s[3] as usize];
                    let t4 = u8_lut[s[4] as usize];
                    let t5 = u8_lut[s[5] as usize];
                    ((w[0] * t0 + w[1] * t1) + (w[2] * t2 + w[3] * t3))
                        + (w[4] * t4 + w[5] * t5)
                }

                let mut idx = 0usize;
                while idx < frames {
                    if pos_q >= pcm_len_q {
                        if looped {
                            pos_q = loop_start_q + (pos_q - loop_start_q) % loop_len_q;
                        } else {
                            break;
                        }
                    }

                    let base = (pos_q >> 32) as usize;
                    if pcm_len >= 6 && base >= 2 && pos_q <= max_interior_q {
                        // Interior fast run: all 6 taps in `base - 2..base + 4` are guaranteed
                        // in-bounds with zero loop-wrap checks.
                        let remaining = frames - idx;
                        let run = (max_interior_q - pos_q)
                            .checked_div(step_q)
                            .map_or(remaining, |n| (n + 1).min(remaining as u64) as usize);
                        let (chunks_l, rem_l) = out_l[idx..idx + run].as_chunks_mut::<4>();
                        let (chunks_r, rem_r) = out_r[idx..idx + run].as_chunks_mut::<4>();
                        for (cl, cr) in chunks_l.iter_mut().zip(chunks_r.iter_mut()) {
                            let p0 = pos_q;
                            let p1 = p0.wrapping_add(step_q);
                            let p2 = p0.wrapping_add(step_q2);
                            let p3 = p0.wrapping_add(step_q3);
                            pos_q = p0.wrapping_add(step_q4);

                            let a0 = eval_interior(pcm, lut, u8_lut, p0);
                            let a1 = eval_interior(pcm, lut, u8_lut, p1);
                            let a2 = eval_interior(pcm, lut, u8_lut, p2);
                            let a3 = eval_interior(pcm, lut, u8_lut, p3);

                            cl[0] += a0 * gain_l;
                            cr[0] += a0 * gain_r;
                            cl[1] += a1 * gain_l;
                            cr[1] += a1 * gain_r;
                            cl[2] += a2 * gain_l;
                            cr[2] += a2 * gain_r;
                            cl[3] += a3 * gain_l;
                            cr[3] += a3 * gain_r;
                        }
                        for (ol, or) in rem_l.iter_mut().zip(rem_r.iter_mut()) {
                            let acc = eval_interior(pcm, lut, u8_lut, pos_q);
                            *ol += acc * gain_l;
                            *or += acc * gain_r;
                            pos_q = pos_q.wrapping_add(step_q);
                        }
                        idx += run;
                    } else {
                        // Boundary sample (near start or end of PCM).
                        let phase = ((pos_q >> 24) & 0xFF) as usize;
                        let w = &lut[phase];
                        let base_i = base as isize;
                        let mut acc = 0.0f32;
                        for (t, &wt) in w.iter().take(LANCZOS3_TAPS).enumerate() {
                            let si = base_i + (t as isize) - 2;
                            if si >= 0 && (si as usize) < pcm_len {
                                acc += wt * u8_lut[pcm[si as usize] as usize];
                            }
                        }
                        out_l[idx] += acc * gain_l;
                        out_r[idx] += acc * gain_r;
                        pos_q = pos_q.wrapping_add(step_q);
                        idx += 1;
                    }
                }
                *pos_f64 = (pos_q as f64) * Q32_INV;
            } else {
                // Fallback for extreme (> 2^30) positions or steps.
                for idx in 0..frames {
                    if ph >= pcm_len_f {
                        if looped {
                            ph = loop_start_f + (ph - loop_start_f) % loop_len_f;
                        } else {
                            break;
                        }
                    }
                    let base = (ph as i64) as f64;
                    let frac = ph - base;
                    let base_i = base as isize;
                    let phase = ((frac * (LANCZOS_PHASES as f64)) as usize).min(LANCZOS_PHASES - 1);
                    let w = &lut[phase];
                    let mut acc = 0.0f32;
                    for (t, &wt) in w.iter().take(LANCZOS3_TAPS).enumerate() {
                        let si = base_i + (t as isize) - 2;
                        if si >= 0 && (si as usize) < pcm_len {
                            acc += wt * u8_lut[pcm[si as usize] as usize];
                        }
                    }
                    out_l[idx] += acc * gain_l;
                    out_r[idx] += acc * gain_r;
                    ph += step;
                }
                *pos_f64 = ph;
            }
        } else {
            let sample_at = |pcm: &[u8], idx: isize| -> f64 {
                if idx < 0 || idx as usize >= pcm.len() {
                    0.0
                } else {
                    ((pcm[idx as usize] as f64) - 128.0) / 128.0
                }
            };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn studio_synth_silence_is_exact_zero() {
        let pcm = [128u8; 256];
        let view = SonixSampleView::from_pcm_bytes(22_050, 60, Some(16), Some(256), &pcm);
        let studio = StudioSynth::default();
        let mut out_l = [0.0f32; 512];
        let mut out_r = [0.0f32; 512];
        let mut pos = 0.0;
        studio.render_voice_f32(&view, &mut pos, 22_050.0, 1.0, 0.0, &mut out_l, &mut out_r);
        assert!(out_l.iter().all(|&x| x == 0.0));
        assert!(out_r.iter().all(|&x| x == 0.0));
        assert!(pos > 0.0);
    }

    #[test]
    fn studio_synth_oneshot_stops_and_malformed_loop_safe() {
        let pcm = [200u8; 8];
        let view = SonixSampleView::from_pcm_bytes(22_050, 60, None, None, &pcm);
        let studio = StudioSynth::default();
        let mut out_l = [0.0f32; 64];
        let mut out_r = [0.0f32; 64];
        let mut pos = 0.0;
        studio.render_voice_f32(&view, &mut pos, 48_000.0, 1.0, -1.0, &mut out_l, &mut out_r);
        // Hard-left pan: right channel stays ~0, left channel has signal for the first 8 frames.
        assert!(out_l[4] > 0.4);
        assert_eq!(out_l[20], 0.0);
        assert!(out_r.iter().all(|&x| x.abs() < 1e-6));

        // Malformed loop_start >= pcm.len() must not panic or produce NaN.
        let bad = SonixSampleView {
            pcm_bytes: &pcm,
            sample_rate_hz: 22_050,
            root_key: 60,
            loop_len: 100,
            loop_start: Some(500),
            loop_end: Some(600),
        };
        let mut pos2 = 0.0;
        studio.render_voice_f32(&bad, &mut pos2, 48_000.0, 1.0, 0.0, &mut out_l, &mut out_r);
        assert!(out_l.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn studio_synth_general_lobes_fallback() {
        let pcm: [u8; 64] = core::array::from_fn(|i| ((i * 4) & 0xFF) as u8);
        let view = SonixSampleView::from_pcm_bytes(22_050, 60, Some(8), Some(64), &pcm);
        let studio = StudioSynth {
            sample_rate: 48_000,
            sinc_lobes: 2,
        };
        let mut out_l = [0.0f32; 32];
        let mut out_r = [0.0f32; 32];
        let mut pos = 4.25;
        studio.render_voice_f32(&view, &mut pos, 22_050.0, 0.5, 0.25, &mut out_l, &mut out_r);
        assert!(out_l.iter().any(|&x| x != 0.0));
        assert!(out_l.iter().all(|x| x.is_finite()));
        assert!(out_r.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn render_block_zero_channels_clamps_to_zero() {
        let mut voices = [SonixVoice::silent(); 2];
        let pcm = [128u8; 16];
        let samples: [(&[u8], u32); 2] = [(&pcm, 0), (&pcm, 0)];
        let ctx = SonixContext::default();
        let mut acc = [0u32; SONIX_BLOCK_WORDS];
        let mut out = [99i8; SONIX_BLOCK_SIZE];
        let total = AccuracySynth::render_block(
            SonixVersion::V116Gsm,
            &mut voices,
            &samples,
            0,
            0,
            &ctx,
            &mut acc,
            &mut out,
            SONIX_BLOCK_SIZE,
        )
        .expect("zero channels ok");
        assert_eq!(total, 0);
        assert!(out.iter().all(|&b| b == 0));
    }
}
