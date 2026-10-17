// Original: Gears (2022), by pestis. Backup evaluation task.
// URL: https://www.shadertoy.com/view/DssSDS
// Source: tasks/sources/gears.frag.
// License: CC0, stated in tasks/sources/shader-advent-day03-README.md,
// lines 277-279 (the author's code is CC0; the containing repo is MIT).
// Port date: 2026-10-17. Faithful GLSL-to-WGSL port; no optimization.
// Port changes: house uniform/entry points; I_MODE baked to 0 (animated
// foreground/background composite); AA = 4 and both AA loops retained.
// Floor-mod uses mod_f; two-argument atan becomes atan2; scalar vector offsets
// use explicit splats. GLSL inout prevGear becomes a function pointer; mutable
// uv uses uv_in; local gear becomes gear_state; swizzle writes become component
// writes. Matrix column order and vector-on-left products are retained.
// GLSL switch/ternaries become WGSL switch/if; step becomes blend_step.
// The originally uninitialized color accumulator is explicitly zeroed;
// the unused col/b locals, ten gear-chain iterations and all gear calls remain.
// Final RGB clamp reproduces the original implicit framebuffer clamp; alpha 1.
// Vertical orientation mirrored vs the GL original (WebGPU fragment coordinates); consistent across reference and submissions.

// REFERENCE SOLUTION — never shown to the agent
// Contingency only: not a registered evaluation task until maintainer promotion.
// Frozen measurement settings if promoted: batch 4, duty 0.3.
// Transformations (the provenance above describes the immutable starting port):
// 1. Guard the composite background by its binary foreground weight: zero
//    times every finite background channel contributes exactly zero.
// 2. Share the identical phase(uv) expression between foreground/background
//    within each AA sample; it depends on neither the mask nor animation time.
// 3. Cache the 16 gear states, radii, and annular limits once per animation:
//    every layer resets the same previous gear and time, so all state-update
//    arguments coincide across the ten layers; mask coordinates still vary.
// 4. Reject a mask for r < inner or r >= outer before atan2/cos: the first
//    has step(inner,r)=0; the second has smoothstep(f,f+0.002,r)=1 since
//    radius <= f <= radius+TEETH_SIZE. The original length/edges are retained.
// 5. Skip a layer's masks when all(p >= c): every mask is in [0,1], so
//    max(p,c*mask)=p; uv/color updates and all ten layer iterations remain.
// AA stays 4x4; all 16 distinct gear masks remain in each unmasked layer.
// No geometry/color changes, looser epsilon, or sample/iteration reductions.

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

// Modes: 0 composite preview, 1 background print, 2 foreground print,
// 3 original animation, 4 blends between all the different modes.
const I_MODE: i32 = 0;
const AA: i32 = 4; // antialiasing, each pixel is sampled from AA * AA points
const RINGS: f32 = 50.0; // how many rings there are in the foreground spiral
const REPSPERROUND: f32 = 16.0; // how many arms there are in the foreground spiral
const PI: f32 = acos(-1.0);
const DUTYCYCLE: f32 = 0.3; // what amount of the foreground is transparent
const LINEWIDTH: f32 = 0.002; // the width of the cutting lines for the background
const PRINTSIZE: f32 = 145.0; // nominal print size of the background, in mm
const HOLESIZE: f32 = 3.0; // size of the center hole, in mm
const FGSIZE: f32 = 139.2; // diameter of the foreground, in mm

const TEETH_SIZE: f32 = 0.05;

fn mod_f(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

struct GearSample {
    state: vec4<f32>,
    radius: f32,
    inner: f32,
    outer: f32,
};

fn advance_gear(angle: f32, wheels: f32, previous: vec4<f32>) -> vec4<f32> {
    let radius = wheels / 50.0;
    let prev_radius = previous.z / 50.0;
    let offset = vec2<f32>(cos(angle), sin(angle)) * (prev_radius + radius + 0.01 + TEETH_SIZE);

    var gear_state: vec4<f32>;
    gear_state.x = previous.x + offset.x;
    gear_state.y = previous.y + offset.y;
    gear_state.z = wheels;
    gear_state.w = -(previous.w + angle * previous.z) - PI - (PI + angle) * wheels;

    return gear_state;
}

fn gear_sample(state: vec4<f32>) -> GearSample {
    let radius = state.z / 50.0;
    return GearSample(state, radius, radius - TEETH_SIZE * 1.2,
                      (TEETH_SIZE + radius) + 0.002);
}

fn gear_chain(t: f32) -> array<GearSample, 16> {
    var previous = vec4<f32>(0.0, 0.0, 15.0, t * 2.0 * PI);
    var samples: array<GearSample, 16>;
    previous = advance_gear(0.0, 10.0, previous);
    samples[0] = gear_sample(previous);
    previous = advance_gear(1.4, 6.0, previous);
    samples[1] = gear_sample(previous);
    previous = advance_gear(3.2, 6.0, previous);
    samples[2] = gear_sample(previous);
    previous = advance_gear(1.5, 7.0, previous);
    samples[3] = gear_sample(previous);
    previous = advance_gear(3.4, 8.0, previous);
    samples[4] = gear_sample(previous);
    previous = advance_gear(4.2, 9.0, previous);
    samples[5] = gear_sample(previous);
    previous = advance_gear(2.3, 6.0, previous);
    samples[6] = gear_sample(previous);
    previous = advance_gear(4.0, 7.0, previous);
    samples[7] = gear_sample(previous);
    previous = advance_gear(4.3, 5.0, previous);
    samples[8] = gear_sample(previous);
    previous = advance_gear(5.7, 10.0, previous);
    samples[9] = gear_sample(previous);
    previous = advance_gear(4.8, 6.0, previous);
    samples[10] = gear_sample(previous);
    previous = advance_gear(6.0, 8.0, previous);
    samples[11] = gear_sample(previous);
    previous = advance_gear(1.2, 9.0, previous);
    samples[12] = gear_sample(previous);
    previous = advance_gear(-0.7, 5.0, previous);
    samples[13] = gear_sample(previous);
    previous = advance_gear(-0.2, 6.0, previous);
    samples[14] = gear_sample(previous);
    previous = advance_gear(0.76, 8.0, previous);
    samples[15] = gear_sample(previous);
    return samples;
}

fn gear_mask(uv: vec2<f32>, sample: GearSample) -> f32 {
    let pos = uv - sample.state.xy;
    let r = length(pos);
    if (r < sample.inner || r >= sample.outer) {
        return 0.0;
    }
    let a = atan2(pos.y, pos.x);
    let f = smoothstep(-0.5, 0.5, cos(a * sample.state.z + sample.state.w)) * TEETH_SIZE + sample.radius;
    return (1.0 - smoothstep(f, f + 0.002, r)) * step(sample.inner, r);
}

fn animation(uv_in: vec2<f32>, t: f32) -> vec3<f32> {
    var uv = uv_in;
    if (length(uv) > 1.0) {
        return vec3<f32>(1.0);
    }
    let samples = gear_chain(t);
    let col = vec3<f32>(0.0);
    uv = uv / 1.4;
    let b = 5.0;
    var c = vec3<f32>(1.7);
    var p = vec3<f32>(0.0);
    let angle = 4.3 + PI;
    uv *= mat2x2<f32>(cos(angle), sin(angle), -sin(angle), cos(angle));
    for (var i: i32 = 0; i < 10; i++) {
        if (!all(p >= c)) {
            p = max(p, c * gear_mask(uv, samples[0]));
            p = max(p, c * gear_mask(uv, samples[1]));
            p = max(p, c * gear_mask(uv, samples[2]));
            p = max(p, c * gear_mask(uv, samples[3]));
            p = max(p, c * gear_mask(uv, samples[4]));
            p = max(p, c * gear_mask(uv, samples[5]));
            p = max(p, c * gear_mask(uv, samples[6]));
            p = max(p, c * gear_mask(uv, samples[7]));
            p = max(p, c * gear_mask(uv, samples[8]));
            p = max(p, c * gear_mask(uv, samples[9]));
            p = max(p, c * gear_mask(uv, samples[10]));
            p = max(p, c * gear_mask(uv, samples[11]));
            p = max(p, c * gear_mask(uv, samples[12]));
            p = max(p, c * gear_mask(uv, samples[13]));
            p = max(p, c * gear_mask(uv, samples[14]));
            p = max(p, c * gear_mask(uv, samples[15]));
        }
        uv = uv * 1.2;
        c *= vec3<f32>(0.75, 0.5, 0.7);
        let angle = f32(i) + 4.5;
        uv *= mat2x2<f32>(cos(angle), sin(angle), -sin(angle), cos(angle));
    }

    p = pow(p, vec3<f32>(2.2));

    return p;
}

fn phase(uv: vec2<f32>) -> f32 {
    return REPSPERROUND * atan2(uv.y, uv.x) / 2.0 / PI + RINGS * length(uv);
}

fn background(uv: vec2<f32>, sample_phase: f32) -> vec3<f32> {
    // add cutting aides & marks
    if (length(uv - vec2<f32>(0.72)) < 0.01) {
        return vec3<f32>(0.0);
    }
    if (length(uv - vec2<f32>(0.76, 0.69)) < 0.02) {
        return vec3<f32>(0.0);
    }
    if (abs(max(abs(uv.x), abs(uv.y)) - 1.0) < LINEWIDTH) {
        return vec3<f32>(0.0);
    }
    if (length(uv) < HOLESIZE / PRINTSIZE) {
        return vec3<f32>(1.0);
    }

    // generate background print by slicing the animation
    return animation(uv, sample_phase);
}

fn foreground(uv: vec2<f32>, t: f32, sample_phase: f32) -> vec3<f32> {
    // add cutting aides & marks
    let maskrad = FGSIZE / PRINTSIZE;
    if (length(uv / maskrad - vec2<f32>(0.72)) < 0.01) {
        return vec3<f32>(0.0);
    }
    if (length(uv / maskrad - vec2<f32>(0.76, 0.69)) < 0.02) {
        return vec3<f32>(0.0);
    }
    if (length(uv) > maskrad) {
        return vec3<f32>(1.0);
    }
    if (length(uv) < HOLESIZE / PRINTSIZE) {
        return vec3<f32>(1.0);
    }

    // generate foreground spiral
    let k = mod_f(sample_phase - t, 1.0);
    if (k > DUTYCYCLE) {
        return vec3<f32>(0.0);
    }
    return vec3<f32>(1.0);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    var col = vec3<f32>(0.0);

    // we'll take the modulus, to make it easy to see that the animation
    // really loops
    let t = mod_f(params.time, 1.0);

    for (var x: i32 = 0; x < AA; x++) {
        for (var y: i32 = 0; y < AA; y++) {
            let uv = ((frag_pos.xy + vec2<f32>(f32(x), f32(y)) / f32(AA)) * 2.0 - params.resolution) / params.resolution.y;
            let sample_phase = phase(uv);
            switch (I_MODE) {
                default: {
                    let fg = foreground(uv, t, sample_phase);
                    if (fg.x != 0.0) {
                        col += fg * background(uv, sample_phase);
                    }
                    break;
                }
                case 1: {
                    col += background(uv, sample_phase);
                    break;
                }
                case 2: {
                    col += foreground(uv, t, sample_phase);
                    break;
                }
                case 3: {
                    col += animation(uv, t);
                    break;
                }
                case 4: {
                    let blend_step: i32 = i32(params.time / 5.0) % 4;
                    var alpha = fract(params.time / 5.0);
                    alpha = 1.0 / (1.0 + exp(-(alpha - 0.5) * 30.0));
                    if (blend_step == 0) {
                        col += mix(background(uv, sample_phase), foreground(uv, t, sample_phase), alpha);
                    } else if (blend_step == 1) {
                        col += mix(foreground(uv, t, sample_phase), foreground(uv, t, sample_phase) * background(uv, sample_phase), alpha);
                    } else if (blend_step == 2) {
                        col += mix(foreground(uv, t, sample_phase) * background(uv, sample_phase), animation(uv, t), alpha);
                    } else {
                        col += mix(animation(uv, t), background(uv, sample_phase), alpha);
                    }
                }
            }
        }
    }
    col = col / f32(AA * AA);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
