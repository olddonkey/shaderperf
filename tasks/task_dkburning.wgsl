// Original: Burning Ship Julia, by darkeclipz (Lars Rotgers).
// URL: ./sources/round2/dk_burningship_julia.js.glsl (local snapshot;
// upstream permalink was not recorded with the supplied source).
// Source: tasks/sources/round2/dk_burningship_julia.js.glsl.
// License: MIT; exact statement: tasks/sources/round2/LICENSE-darkeclipz,
// lines 1-21, copyright (c) 2018 Lars Rotgers.
// Port date: 2026-10-17. Faithful GLSL-to-WGSL port; no optimization.
// Port changes: GLSL extracted from the JS template string, lines 2-93;
// wrapper on lines 1/94 discarded. House uniform/entry points replace the
// GLSL uniforms, varying UV-to-pixel conversion, and output assignment.
// AA = 32 jitter samples and MAX_ITER = 1024 retained, including the original
// post-update escape test and i=j only after non-escaping iterations.
// Original random/nextRand2 hash and per-invocation seed retained; the two
// post-increments are sequenced left-to-right, yielding (0,1), (2,3), ... .
// Matrix columns and scalar/vector splats are explicit.
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

const AA: f32 = 32.0;
const MAX_ITER: f32 = 1024.0;
const THRESHOLD: f32 = 64.0;
const LN2: f32 = 0.6931471806;

struct palette {
    c0: vec3<f32>,
    c1: vec3<f32>,
    c2: vec3<f32>,
    c3: vec3<f32>,
    c4: vec3<f32>,
};

fn palette_blue() -> palette {
    var p: palette;
    p.c0 = vec3<f32>(0.0, 2.0, 5.0) / 255.0;
    p.c1 = vec3<f32>(8.0, 45.0, 58.0) / 255.0;
    p.c2 = vec3<f32>(38.0, 116.0, 145.0) / 255.0;
    p.c3 = vec3<f32>(167.0, 184.0, 181.0) / 255.0;
    p.c4 = vec3<f32>(207.0, 197.0, 188.0) / 255.0;
    return p;
}

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

fn bship(uv: vec2<f32>) -> f32 {
    let c = vec2<f32>(0.5971, -0.9370);
    var z = uv;
    var i = 0.0;
    for (var j: f32 = 0.0; j < MAX_ITER; j += 1.0) {
        // z = (|Re(z)| + i|Im(z)|)^2
        z = abs(z);
        z = mat2x2<f32>(z, vec2<f32>(-z.y, z.x)) * z + c;
        if (dot(z, z) > THRESHOLD) { break; }
        i = j;
    }
    return i - log(log(dot(z, z)) / LN2) / LN2;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let R = params.resolution;
    let pal = palette_blue();
    var col = vec3<f32>(0.0);
    let a = 3.141592654 / 2.0;
    let r = mat2x2<f32>(vec2<f32>(cos(a), sin(a)), vec2<f32>(-sin(a), cos(a)));
    for (var i: f32 = 0.0; i < AA; i += 1.0) {
        let p = 1.6 * r * ((2.0 * frag_pos.xy - R + nextRand2() + vec2<f32>(0.0001)) / R.y);
        let orbit = bship(p) / MAX_ITER;
        let f = fract(10.0 * orbit);
        col += cmap(fract(f + params.time / 10.0) - 0.0001, pal);
    }
    col /= AA;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
