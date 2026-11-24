//! Byte-exact ARM7TDMI golden tests for Sonix v1.09c, v1.096+GSM, and v1.16+GSM.
//!
//! Every golden `.bin` fixture in `tests/golden/` was captured by executing the real
//! ARM7TDMI Sonix mixer routines from retail ROMs in headless `libmgba.so.0.10` using
//! clean-room synthetic deterministic waveforms (zero copyrighted game audio data).
//! Fixture layout: 128 DMA output bytes, then `total_volume: u16`, `phase_hi_byte: u8`,
//! padding `u8`, `phase_lo: u32`, and `env_level: u16` (all little-endian). Every byte
//! is verified with strict `assert_eq!` (0 tolerance).

use cosmigo_sonix::{
    AccuracySynth, SonixContext, SonixVersion, SonixVoice, SONIX_BLOCK_SIZE, SONIX_BLOCK_WORDS,
};

fn make_ramp() -> [u8; 1024] {
    core::array::from_fn(|i| ((i * 7 + 3) & 0xFF) as u8)
}

fn make_extrema() -> [u8; 1024] {
    core::array::from_fn(|i| if i % 2 == 0 { 0 } else { 255 })
}

fn ctx(shift: u8) -> SonixContext {
    SonixContext {
        master_a: 200,
        master_b: 100,
        master_c: 255,
        step_shift: shift,
    }
}

fn parse_fixture(golden: &[u8]) -> (&[u8], u16, u8, u32, u16) {
    assert_eq!(golden.len(), 138, "fixture length");
    let dma = &golden[0..128];
    let total = u16::from_le_bytes([golden[128], golden[129]]);
    let phase_hi_byte = golden[130];
    let phase_lo = u32::from_le_bytes([golden[132], golden[133], golden[134], golden[135]]);
    let env_level = u16::from_le_bytes([golden[136], golden[137]]);
    (dma, total, phase_hi_byte, phase_lo, env_level)
}

fn run_single(
    version: SonixVersion,
    pcm: &[u8],
    loop_len: u32,
    voice: &mut SonixVoice,
    shift: u8,
) -> ([i8; SONIX_BLOCK_SIZE], u16) {
    let mut acc = [0u32; SONIX_BLOCK_WORDS];
    let mut out = [0i8; SONIX_BLOCK_SIZE];
    let eff = AccuracySynth::mix_channel(
        version,
        &mut acc,
        pcm,
        loop_len,
        voice,
        &ctx(shift),
        false,
        SONIX_BLOCK_SIZE,
    );
    let total = eff as u16;
    AccuracySynth::clamp_block(version, &acc, total, &mut out);
    (out, total)
}

fn check(
    version: SonixVersion,
    golden: &[u8],
    out: &[i8; SONIX_BLOCK_SIZE],
    total: u16,
    voice: &SonixVoice,
) {
    let (dma, g_total, g_phb, g_phlo, g_env) = parse_fixture(golden);
    for (i, &byte) in out.iter().enumerate() {
        assert_eq!(
            byte, dma[i] as i8,
            "{version}: DMA byte mismatch at index {i}"
        );
    }
    assert_eq!(total, g_total, "total volume mismatch");
    assert_eq!(voice.phase_hi_byte, g_phb, "phase hi byte mismatch");
    assert_eq!(voice.phase_lo, g_phlo, "phase_lo mismatch");
    assert_eq!(voice.env_level, g_env, "env_level mismatch");
}

fn base_voice(step: u32) -> SonixVoice {
    let mut v = SonixVoice::new_simple(64, 200, step);
    v.env_level = 200 << 8;
    v
}

#[test]
fn golden_basic_step1_all_versions() {
    let pcm = make_ramp();
    for (version, golden) in [
        (
            SonixVersion::V109c,
            include_bytes!("golden/v109c_basic_step1.bin").as_slice(),
        ),
        (
            SonixVersion::V1096Gsm,
            include_bytes!("golden/v1096_basic_step1.bin").as_slice(),
        ),
        (
            SonixVersion::V116Gsm,
            include_bytes!("golden/v116_basic_step1.bin").as_slice(),
        ),
    ] {
        let mut voice = base_voice(0x0001_0000);
        let (out, total) = run_single(version, &pcm, 0, &mut voice, 10);
        check(version, golden, &out, total, &voice);
    }
}

#[test]
fn golden_fractional_step_all_versions() {
    let pcm = make_ramp();
    for (version, golden) in [
        (
            SonixVersion::V109c,
            include_bytes!("golden/v109c_fractional_step.bin").as_slice(),
        ),
        (
            SonixVersion::V1096Gsm,
            include_bytes!("golden/v1096_fractional_step.bin").as_slice(),
        ),
        (
            SonixVersion::V116Gsm,
            include_bytes!("golden/v116_fractional_step.bin").as_slice(),
        ),
    ] {
        let mut voice = base_voice(0x0001_8000);
        let (out, total) = run_single(version, &pcm, 0, &mut voice, 10);
        check(version, golden, &out, total, &voice);
    }
}

#[test]
fn golden_loop_wrap_all_versions() {
    let pcm = make_ramp();
    for (version, golden) in [
        (
            SonixVersion::V109c,
            include_bytes!("golden/v109c_loop_wrap.bin").as_slice(),
        ),
        (
            SonixVersion::V1096Gsm,
            include_bytes!("golden/v1096_loop_wrap.bin").as_slice(),
        ),
        (
            SonixVersion::V116Gsm,
            include_bytes!("golden/v116_loop_wrap.bin").as_slice(),
        ),
    ] {
        let mut voice = base_voice(0x0001_0000);
        voice.phase_hi_byte = 0xFF;
        voice.phase_lo = 0xF9C << 20;
        // The harness aligned the sample end at phase 0x100000; mirror that
        // here with a 1 MiB phase space whose last 100 bytes are the ramp.
        let mut phase_space = vec![0u8; 0x10_0000];
        phase_space[0xFFF9C..0xFFF9C + 100].copy_from_slice(&pcm[..100]);
        let (out, total) = run_single(version, &phase_space, 60, &mut voice, 10);
        check(version, golden, &out, total, &voice);
    }
}

#[test]
fn golden_two_channel_all_versions() {
    let pcm = make_ramp();
    for (version, golden) in [
        (
            SonixVersion::V109c,
            include_bytes!("golden/v109c_two_channel.bin").as_slice(),
        ),
        (
            SonixVersion::V1096Gsm,
            include_bytes!("golden/v1096_two_channel.bin").as_slice(),
        ),
        (
            SonixVersion::V116Gsm,
            include_bytes!("golden/v116_two_channel.bin").as_slice(),
        ),
    ] {
        let mut voices = [SonixVoice::silent(); 5];
        voices[3] = base_voice(0x0001_0000);
        voices[3].next_index = 4;
        voices[4] = SonixVoice::new_simple(200, 180, 0x0000_C000);
        voices[4].env_level = 180 << 8;
        let samples: [(&[u8], u32); 5] = [(&pcm, 0), (&pcm, 0), (&pcm, 0), (&pcm, 0), (&pcm, 0)];
        let mut acc = [0u32; SONIX_BLOCK_WORDS];
        let mut out = [0i8; SONIX_BLOCK_SIZE];
        let total = AccuracySynth::render_block(
            version,
            &mut voices,
            &samples,
            3,
            2,
            &ctx(10),
            &mut acc,
            &mut out,
            SONIX_BLOCK_SIZE,
        )
        .expect("render_block");
        // Fixture phase/env fields describe channel 3 (the first channel).
        check(version, golden, &out, total, &voices[3]);
    }
}

#[test]
fn golden_envelope_all_versions() {
    let pcm = make_ramp();
    for (version, golden) in [
        (
            SonixVersion::V109c,
            include_bytes!("golden/v109c_envelope.bin").as_slice(),
        ),
        (
            SonixVersion::V1096Gsm,
            include_bytes!("golden/v1096_envelope.bin").as_slice(),
        ),
        (
            SonixVersion::V116Gsm,
            include_bytes!("golden/v116_envelope.bin").as_slice(),
        ),
    ] {
        let mut voice = SonixVoice::new_simple(64, 200, 0x0040_0000);
        voice.flags = 0x10;
        voice.env_level = 0xC8E8;
        voice.env_rate = 10;
        let (out, total) = run_single(version, &pcm, 0, &mut voice, 4);
        check(version, golden, &out, total, &voice);
    }
}

#[test]
fn golden_extrema_all_versions() {
    let pcm = make_extrema();
    for (version, golden) in [
        (
            SonixVersion::V109c,
            include_bytes!("golden/v109c_extrema.bin").as_slice(),
        ),
        (
            SonixVersion::V1096Gsm,
            include_bytes!("golden/v1096_extrema.bin").as_slice(),
        ),
        (
            SonixVersion::V116Gsm,
            include_bytes!("golden/v116_extrema.bin").as_slice(),
        ),
    ] {
        let mut voice = SonixVoice::new_simple(255, 255, 0x0001_0000);
        voice.env_level = 255 << 8;
        let (out, total) = run_single(version, &pcm, 0, &mut voice, 10);
        check(version, golden, &out, total, &voice);
    }
}

#[test]
fn golden_past_end_all_versions() {
    let pcm = make_ramp();
    for (version, golden) in [
        (
            SonixVersion::V109c,
            include_bytes!("golden/v109c_past_end.bin").as_slice(),
        ),
        (
            SonixVersion::V1096Gsm,
            include_bytes!("golden/v1096_past_end.bin").as_slice(),
        ),
        (
            SonixVersion::V116Gsm,
            include_bytes!("golden/v116_past_end.bin").as_slice(),
        ),
    ] {
        let mut voice = base_voice(0x0002_0000);
        voice.phase_lo = 200 << 20;
        // The harness PCM region is 1024 ramp bytes followed by zeros in the
        // image; this test's 1024-byte ramp matches the first 56 fetches and
        // out-of-range fetches (index >= 1024) return 0 in both models. The
        // harness reads image bytes past 1024 up to the DMA area, which are
        // zero, matching the Rust out-of-range zero.
        let (out, total) = run_single(version, &pcm, 0, &mut voice, 10);
        check(version, golden, &out, total, &voice);
    }
}
