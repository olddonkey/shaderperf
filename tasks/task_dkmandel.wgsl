// Original: Mandelbrot, by darkeclipz (Lars Rotgers).
// URL: ./sources/round2/dk_mandelbrot.glsl (local snapshot;
// upstream permalink was not recorded with the supplied source).
// Source: tasks/sources/round2/dk_mandelbrot.glsl.
// License: MIT; exact statement: tasks/sources/round2/LICENSE-darkeclipz,
// lines 1-21, copyright (c) 2018 Lars Rotgers.
// Port date: 2026-10-17. Faithful GLSL-to-WGSL port; no optimization.
// Port changes: house uniform/entry points; SAMPLES = 3 jitter AA retained;
// MAX_ITER = 450 (the commented 1024 alternative remains inactive).
// Original random/nextRand2 hash and per-invocation seed retained; the two
// post-increments are sequenced left-to-right, yielding (0,1), (2,3), ... .
// GLSL pre-increment loop condition is expanded, including its final increment.
// Matrix columns and scalar/vector splats are explicit; unused palettes kept.
// Final RGB is clamped to [0,1] to match framebuffer clamp; alpha is 1.0.
// Vertical orientation mirrored vs the GL original (WebGPU fragment coordinates); consistent across reference and submissions. No Y flip.

struct Params { resolution: vec2<f32>, time: f32, _pad: f32, };
@group(0) @binding(0) var<uniform> params: Params;

struct VSOut { @builtin(position) pos: vec4<f32>, };

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var out: VSOut;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

const LN2: f32 = 0.6931471806;

struct palette {
    c0: vec3<f32>,
    c1: vec3<f32>,
    c2: vec3<f32>,
    c3: vec3<f32>,
    c4: vec3<f32>,
};

fn palette_red_accented() -> palette {
    var p: palette;
    p.c0 = vec3<f32>(0.0, 0.0, 16.0) / 255.0;
    p.c1 = vec3<f32>(202.0, 202.0, 200.0) / 255.0;
    p.c2 = vec3<f32>(255.0) / 255.0;
    p.c3 = vec3<f32>(182.0, 132.0, 107.0) / 255.0;
    p.c4 = vec3<f32>(247.0, 49.0, 73.0) / 255.0;
    return p;
}

fn palette_calm_blue() -> palette {
    var p: palette;
    p.c0 = vec3<f32>(5.0, 8.0, 49.0) / 255.0;
    p.c1 = vec3<f32>(19.0, 48.0, 114.0) / 255.0;
    p.c2 = vec3<f32>(213.0, 230.0, 247.0) / 255.0;
    p.c3 = vec3<f32>(213.0, 230.0, 247.0) / 255.0;
    p.c4 = vec3<f32>(5.0, 8.0, 49.0) / 255.0;
    return p;
}

fn palette_green() -> palette {
    var p: palette;
    p.c0 = vec3<f32>(0.0, 120.0, 122.0) / 255.0;
    p.c1 = vec3<f32>(148.0, 235.0, 216.0) / 255.0;
    p.c2 = vec3<f32>(0.0, 179.0, 73.0) / 255.0;
    p.c3 = vec3<f32>(0.0, 121.0, 57.0) / 255.0;
    p.c4 = vec3<f32>(0.0, 61.0, 52.0) / 255.0;
    return p;
}

fn palette_blue() -> palette {
    var p: palette;
    p.c0 = vec3<f32>(0.0, 2.0, 5.0) / 255.0;
    p.c1 = vec3<f32>(8.0, 45.0, 58.0) / 255.0;
    p.c2 = vec3<f32>(38.0, 116.0, 145.0) / 255.0;
    p.c3 = vec3<f32>(167.0, 184.0, 181.0) / 255.0;
    p.c4 = vec3<f32>(207.0, 197.0, 188.0) / 255.0;
    return p;
}

// Random: original hash and seed progression, not a replacement RNG.
fn random(st: vec2<f32>) -> f32 {
    return fract(sin(dot(st.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

var<private> randSeed: f32 = 0.0;

fn nextRand2() -> vec2<f32> {
    let vx = randSeed;
    randSeed += 1.0;
    let vy = randSeed;
    randSeed += 1.0;
    let v = vec2<f32>(vx, vy);
    return vec2<f32>(random(v + vec2<f32>(0.34)), random(v + vec2<f32>(0.75)));
}

fn cmap(t: f32, p: palette) -> vec3<f32> {
    var col = vec3<f32>(0.0);
    col = mix(p.c0, p.c1, smoothstep(0.0, 0.2, t));
    col = mix(col, p.c2, smoothstep(0.2, 0.4, t));
    col = mix(col, p.c3, smoothstep(0.4, 0.6, t));
    col = mix(col, p.c4, smoothstep(0.6, 0.8, t));
    col = mix(col, p.c0, smoothstep(0.8, 1.0, t));
    return col;
}

// Inactive source alternative: MAX_ITER = 1024.0.
const MAX_ITER: f32 = 450.0;
const THRESHOLD: f32 = 4.0;

fn mandelbrot(uv: vec2<f32>) -> f32 {
    let c = 2.5 * (uv - vec2<f32>(0.2, 0.0));
    var z = vec2<f32>(0.0);
    var i = 0.0;
    loop {
        i += 1.0;
        if (i > MAX_ITER) { break; }
        z = mat2x2<f32>(z, vec2<f32>(-z.y, z.x)) * z + c;
        if (dot(z, z) > THRESHOLD) { break; }
    }
    return i - log(log(dot(z, z)) / LN2) / LN2;
}

const SAMPLES: f32 = 3.0;

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let R = params.resolution;
    let pal = palette_blue();
    var col = vec3<f32>(0.0);
    for (var i: f32 = 0.0; i < SAMPLES; i += 1.0) {
        let p = 0.00017 * (2.0 * frag_pos.xy - R + nextRand2()) / R.y - vec2<f32>(-0.35209, 0.09199);
        let orbit = mandelbrot(p) / MAX_ITER;
        col += cmap(fract(3.3 * orbit + params.time / 8.0) - 0.0001, pal);
    }
    col /= SAMPLES;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
