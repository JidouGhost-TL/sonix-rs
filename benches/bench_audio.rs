use std::hint::black_box;
use std::time::Instant;

use sonix::{
    AccuracySynth, SonixContext, SonixSampleView, SonixVersion, SonixVoice, StudioSynth,
    SONIX_BLOCK_SIZE, SONIX_BLOCK_WORDS,
};

fn main() {
    println!("=== sonix-rs Audio Performance Benchmarks ===");

    let pcm_a: Vec<u8> = (0i32..4096).map(|i| ((i * 19 + 7) % 256) as u8).collect();
    let pcm_b: Vec<u8> = (0i32..4096).map(|i| ((i * 31 + 53) % 256) as u8).collect();
    let sample = SonixSampleView::from_pcm_bytes(22_050, 60, Some(512), Some(4096), &pcm_a);
    let ctx = SonixContext::default();

    let iters = 500_000usize;

    // 1. AccuracySynth::mix_channel (128 frames, store path)
    let mut acc = [0u32; SONIX_BLOCK_WORDS];
    let start = Instant::now();
    for _ in 0..iters {
        let mut voice = SonixVoice::new_simple(64, 200, 0x0001_0000);
        AccuracySynth::mix_channel(
            SonixVersion::V116Gsm,
            black_box(&mut acc),
            black_box(&pcm_a),
            0,
            &mut voice,
            &ctx,
            false,
            SONIX_BLOCK_SIZE,
        );
        black_box(&acc);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::mix_channel store (128f):     {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * SONIX_BLOCK_SIZE) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 2. AccuracySynth::mix_channel (128 frames, accumulate path)
    let start = Instant::now();
    for _ in 0..iters {
        let mut voice = SonixVoice::new_simple(200, 180, 0x0001_8000);
        AccuracySynth::mix_channel(
            SonixVersion::V116Gsm,
            black_box(&mut acc),
            black_box(&pcm_b),
            0,
            &mut voice,
            &ctx,
            true,
            SONIX_BLOCK_SIZE,
        );
        black_box(&acc);
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::mix_channel accum (128f):     {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * SONIX_BLOCK_SIZE) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 3. AccuracySynth::clamp_block (128 frames)
    let mut out = [0i8; SONIX_BLOCK_SIZE];
    let start = Instant::now();
    for _ in 0..iters {
        AccuracySynth::clamp_block(SonixVersion::V116Gsm, black_box(&acc), 300, black_box(&mut out));
    }
    let elapsed = start.elapsed();
    println!(
        "AccuracySynth::clamp_block (128f):           {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / iters as f64,
        ((iters * SONIX_BLOCK_SIZE) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );

    // 4. StudioSynth 48 kHz Lanczos-3 (1024 frames)
    let studio = StudioSynth::default();
    let mut out_l = vec![0.0f32; 1024];
    let mut out_r = vec![0.0f32; 1024];
    let studio_iters = 50_000usize;
    let start = Instant::now();
    for _ in 0..studio_iters {
        let mut pos = 0.0f64;
        studio.render_voice_f32(
            black_box(&sample),
            &mut pos,
            22_050.0,
            0.85,
            0.25,
            &mut out_l,
            &mut out_r,
        );
        black_box((&out_l, &out_r));
    }
    let elapsed = start.elapsed();
    println!(
        "StudioSynth::render_voice_f32 (48kHz L3, 1k): {:8.2} ns/iter ({:8.2} MSamples/sec)",
        elapsed.as_nanos() as f64 / studio_iters as f64,
        ((studio_iters * 1024) as f64 / elapsed.as_secs_f64()) / 1_000_000.0
    );
}
