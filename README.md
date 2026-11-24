# sonix-rs

**Zero-dependency, pure Rust parser, software synthesizer, and byte-exact golden-tested emulator for the Cosmigo Sonix Sound System (Game Boy Advance).**

[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)
[![no_std](https://img.shields.io/badge/no__std-compatible-success.svg)](#features)

The **Cosmigo Sonix Sound System** powered **14 commercial Game Boy Advance ROMs** including *Ultimate Brain Games*, *Ultimate Card Games* (3 revisions), *Tron 2.0: Killer App* (USA & Europe), and eight 2-in-1 game packs (Shrek 2 + Shark Tale, Spider-Man + X2, Tony Hawk's Underground + Kelly Slater's Pro Surfer, Shrek 2 + Madagascar, and regional variants).

`sonix` provides a clean-room, reverse-engineered implementation designed for audio preservation, DAWs, PC source ports, and emulators.

> **Looking for a ready-to-use CLI?** You can scan GBA ROMs, extract WAV samples, and compile SoundFont 2.04 (.sf2) banks directly from the command line using our **[gbakit](https://github.com/JidouGhost-TL/gbakit)** toolkit (`gbakit audio scan`, `gbakit audio export`, `gbakit audio soundfont`).

---

## Features

- **Zero External Dependencies**: Pure Rust with no dependencies in `[dependencies]`, `#![deny(unsafe_code)]`.
- **`#![no_std]` Compatible**: Ready for embedded targets and WASM runtimes (`alloc` optional).
- **Dual Synthesis Engines**:
  - **`AccuracySynth`**: 100% bit-exact emulation of all three Sonix ARM7TDMI releases — including the 64-bit phase accumulator (20-bit integer + fraction), unsigned 8-bit samples, packed `(s * vol) >> shift` dual-sample scaling with store/accumulate mask quirks, volume-envelope decay, loop wrap by repeated loop-length subtraction on phase carry, and the DC-bias saturating clamp.
  - **`StudioSynth`**: Renders at 48,000 Hz, 32-bit floating-point precision with a compile-time 256-phase × 6-tap Lanczos-3 polyphase windowed-sinc LUT (plus general Lanczos-`a` fallback) and true equal-power stereo panning.
- **Byte-Exact `libmgba` ARM Golden Tested**: 2,688 / 2,688 DMA output bytes plus updated voice phase, envelope, and total-volume state verified strictly byte-for-byte (`assert_eq!`) across all three Sonix releases.

---

## Supported Sonix Engine Revisions

Auditing all 3,377 GBA ROMs identifies **3 distinct releases** spanning **14 commercial ROMs**, all detected via `SonixVersion::detect_from_bytes` and byte-exact golden-tested:

| Revision | Corpus Share | Representative Titles | Mix Shift / Masks | Accumulator Offset | GSM Codec |
| --- | ---: | --- | --- | ---: | --- |
| `V109c` | 1 ROM | *Ultimate Brain Games* | `>>2`, store `0xFFFF3FFF`, accum `bic 0x30000`, clamp `>>21` | `0x358` | No |
| `V1096Gsm` | 11 ROMs | *Ultimate Card Games* (3 revs), eight 2-in-1 packs | `>>3`, store `0xFFFF1FFF`, accum `bic 0x70000`, clamp `>>20` | `0x358` | Yes |
| `V116Gsm` | 2 ROMs | *Tron 2.0: Killer App* (USA, Europe) | `>>3`, store `0xFFFF1FFF`, accum `bic 0x70000`, clamp `>>20` | `0x35C` | Yes |

The GSM voice codec bundled with the later releases is a separate streamed-speech path and is not part of the block sample mixer emulated here.

---

## Quickstart

### Zero-Allocation Mixing (`#![no_std]`)

```rust
use sonix::{AccuracySynth, SonixContext, SonixVersion, SonixVoice, SONIX_BLOCK_SIZE, SONIX_BLOCK_WORDS};

let pcm: &[u8] = &[128, 160, 192, 224, 255, 224, 192, 160, 128, 96, 64, 32, 0, 32, 64, 96];
let ctx = SonixContext::default();
let mut voice = SonixVoice::new_simple(64, 200, 0x0001_0000);
let mut acc = [0u32; SONIX_BLOCK_WORDS];
let mut out = [0i8; SONIX_BLOCK_SIZE];

let total = AccuracySynth::mix_channel(SonixVersion::V116Gsm, &mut acc, pcm, 0, &mut voice, &ctx, false, SONIX_BLOCK_SIZE);
AccuracySynth::clamp_block(SonixVersion::V116Gsm, &acc, total as u16, &mut out);
```

---

## Golden Test Methodology

Every mixing path in `AccuracySynth` is verified strictly byte-for-byte (`assert_eq!`, zero tolerance) against real ARM7TDMI execution traces captured in headless `libmgba.so.0.10` using clean-room synthetic deterministic waveforms:

- **7 test vectors × 3 revisions (2,688 DMA bytes + phase/envelope/total state)**: 1:1 pitch, fractional resampling (1.5× step), loop wrap at the 1 MiB phase boundary, two-channel store + accumulate mixing, volume-envelope decay, alternating-extrema saturation, and past-end zero-fill behaviour.

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))
