// Original: Mandel Lobes, by Orion Elenzil (2019-05-21).
// URL: ./sources/round2/elenzil_mandel_lobes.glsl (local snapshot;
// upstream permalink was not recorded; source line 2 cites inspiration,
// https://www.shadertoy.com/view/wtfGWS, rather than identifying this shader).
// Source: tasks/sources/round2/elenzil_mandel_lobes.glsl.
// License: MIT; exact statement: tasks/sources/round2/LICENSE-elenzil,
// lines 1-21, copyright (c) 2019 Orion Elenzil.
// Port date: 2026-10-17. Faithful GLSL-to-WGSL port; no optimization.
// Port changes: house uniform/entry points; AA = 2.0, both AA loops retained.
// iMouse.xy baked to vec2(0,0); original mouse branch kept with that value.
// Out cycle lengths become function pointers; mutable arguments use copies.
// poi2 remains selected; unused poi1 and t stay unused; complexMul unchanged.
// GRIMOIRE wrapper omitted; vector clamps use explicit splats.
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

const PI: f32 = 3.14159265359;
const TAU: f32 = PI * 2.0;
const AA: f32 = 2.0;
const MOUSE: vec2<f32> = vec2<f32>(0.0);

// Source TODOs remain unimplemented: use a matrix / replace with complexSqr.
fn complexMul(A: vec2<f32>, B: vec2<f32>) -> vec2<f32> {
    return vec2<f32>((A.x * B.x) - (A.y * B.y), (A.x * B.y) + (A.y * B.x));
}

// https://github.com/hughsk/glsl-hsv2rgb/blob/master/index.glsl
fn hsv2rgb(c: vec3<f32>) -> vec3<f32> {
    let K = vec4<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    let p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, vec3<f32>(0.0), vec3<f32>(1.0)), c.y);
}

struct POI {
    center: vec2<f32>,
    range: f32,
    maxIter: f32,
};

fn mandelEscapeIters(C: vec2<f32>, maxIter: f32, ocOff: vec2<f32>,
                     cycleLength1: ptr<function, f32>, cycleLength2: ptr<function, f32>) -> f32 {
    var Z = C;
    var orbitCenter1 = C + 0.30 * vec2<f32>(cos(params.time * 1.30), sin(params.time * 1.00));
    var orbitCenter2 = C + (sin(params.time * 0.2) * 0.1 + 0.1) * vec2<f32>(cos(params.time * 2.30), sin(params.time * 2.30));
    orbitCenter1 += ocOff;
    orbitCenter2 += ocOff;
    *cycleLength1 = 0.0;
    *cycleLength2 = 0.0;
    for (var n: f32 = 0.0; n < maxIter; n += 1.0) {
        Z = complexMul(Z, Z) + C;
        if (*cycleLength1 == 0.0 && abs(1.0 - length(Z - orbitCenter1)) < 0.015) {
            *cycleLength1 = n;
        }
        if (*cycleLength2 == 0.0 && abs(sin(params.time * 0.1) * 0.05 + 0.055 - length(Z - orbitCenter2)) < 0.02) {
            *cycleLength2 = n;
        }
        if (dot(Z, Z) > 4.0) {
            return n;
        }
    }
    return maxIter;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let smallWay = min(params.resolution.x, params.resolution.y);
    let uv = (frag_pos.xy * 2.0 - params.resolution) / smallWay;
    let t = params.time * TAU / 5.0;
    var ocOff = vec2<f32>(0.0);
    if (any(MOUSE > vec2<f32>(50.0))) {
        ocOff = (MOUSE * 2.0 - params.resolution) / smallWay;
    }
    let poi1 = POI(vec2<f32>(-0.7105, 0.2466), 0.04, 90.0);
    let poi2 = POI(vec2<f32>(-0.600, 0.0000), 1.200, 70.0);
    let poi = poi2;
    let rng = poi.range; // Source's optional animated range remains inactive.
    var cycleLength1: f32;
    var cycleLength2: f32;
    var col = vec3<f32>(0.0);
    for (var m: f32 = 0.0; m < AA; m += 1.0) {
        for (var n: f32 = 0.0; n < AA; n += 1.0) {
            let C = (uv + vec2<f32>(m, n) / (AA * smallWay)) * rng + poi.center;
            let escapeIters = mandelEscapeIters(C, poi.maxIter, ocOff, &cycleLength1, &cycleLength2);
            var f = escapeIters / poi.maxIter;
            if (escapeIters == poi.maxIter) { f = 0.0; }
            f = pow(f, 0.6);
            f *= 0.82;
            var rgb = vec3<f32>(f * 0.2, f * 0.6, f * 1.0);
            if (cycleLength1 > 0.0) {
                let spectrumSamples = 20.0;
                rgb += vec3<f32>(cos(cycleLength1 / spectrumSamples * TAU) * 0.2 + 0.3);
            }
            if (cycleLength2 > 0.0) {
                let spectrumSamples = 30.0;
                rgb += hsv2rgb(vec3<f32>(fract(cycleLength2 / spectrumSamples), 0.9, 0.8));
            }
            col += rgb;
        }
    }
    col /= AA * AA;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
