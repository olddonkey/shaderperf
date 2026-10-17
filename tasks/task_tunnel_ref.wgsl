// Original: Tunnel (2021), by pestis.
// URL: https://www.shadertoy.com/view/NtKGWh
// Source: tasks/sources/tunnel.frag.
// License: CC0, stated in tasks/sources/shader-advent-day03-README.md,
// lines 277-279 (the author's code is CC0; the containing repo is MIT).
// Port date: 2026-10-17. Faithful GLSL-to-WGSL port; no optimization.
// Port changes: house uniform/entry points; I_MODE baked to 0 (animated
// foreground/background composite); AA = 4 and both AA loops retained.
// Floor-mod uses mod_f/mod_v2; two-argument atan becomes atan2; scalar
// vector offsets use explicit splats; mutable parameters p/x use p_in/x_in.
// GLSL switch/ternaries become WGSL switch/if; step becomes blend_step.
// The originally uninitialized color accumulator is explicitly zeroed.
// Final RGB clamp reproduces the original implicit framebuffer clamp; alpha 1.
// Vertical orientation mirrored vs the GL original (WebGPU fragment coordinates); consistent across reference and submissions.

// REFERENCE SOLUTION — never shown to the agent
// Transformations (the provenance above describes the immutable starting port):
// 1. Guard the composite background by its binary foreground weight: a zero
//    weight makes every finite background channel contribute exactly zero.
// 2. Cache five Voronoi sites per query: the nine neighbors use only the five
//    integer sums i+j; identical n+g2 expressions give identical hashes/offsets.
// 3. Defer map color until a hit: non-hit iterations consume only distance;
//    the hit retains the same u, v, q.x and the original color expression.
// AA remains 4x4, the march cap remains 70, and the hit epsilon stays 0.001.

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

fn mod_f(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn mod_v2(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    return x - y * floor(x / y);
}

fn hash(p_in: vec2<f32>) -> vec2<f32> {
    var p = p_in;
    p = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)),
                  dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(p) * 18.5453);
}

struct VoronoiSite { id: vec2<f32>, offset: vec2<f32>, };

fn voronoi_site(n: vec2<f32>, diagonal: f32) -> VoronoiSite {
    let o = hash(mod_v2(n + vec2<f32>(diagonal, 0.0), vec2<f32>(12.0)));
    return VoronoiSite(o, vec2<f32>(0.5) + 0.5 * sin(3.0 * o - vec2<f32>(0.1)));
}

// return distance, and cell id
fn voronoi(x_in: vec2<f32>) -> vec2<f32> {
    var x = x_in;
    x.x = mod_f(x.x * 12.0, 12.0);
    let n = vec2<f32>(floor(x.x) + floor(x.y), 0.0);
    let f = fract(x);
    let sites = array<VoronoiSite, 5>(
        voronoi_site(n, -2.0), voronoi_site(n, -1.0), voronoi_site(n, 0.0),
        voronoi_site(n, 1.0), voronoi_site(n, 2.0));

    var m = vec3<f32>(8.0);
    for (var j: i32 = -1; j <= 1; j++) {
        for (var i: i32 = -1; i <= 1; i++) {
            let g = vec2<f32>(f32(i), f32(j));
            let site = sites[u32(i + j + 2)];
            let o = site.id;
            let r = g - f + site.offset;
            let d = dot(r, r);
            if (d < m.x) {
                m = vec3<f32>(d, o);
            }
        }
    }

    return vec2<f32>(sqrt(m.x), m.y + m.z);
}

// March data: u, v, Voronoi distance, and signed march distance.
fn map(p: vec3<f32>, t: f32) -> vec4<f32> {
    let u = atan2(p.y, p.x) / PI;
    let v = p.z * 4.0;
    let q = voronoi(vec2<f32>(u - t / 12.0, v + t));
    return vec4<f32>(u, v, q.x, 1.0 - length(p.xy) - q.x / 3.0);
}

fn map_color(m: vec4<f32>) -> vec3<f32> {
    var col = vec3<f32>(0.6) + 0.4 * cos(vec3<f32>(m.x * PI) + vec3<f32>(0.0, 0.5, 1.0) + vec3<f32>(m.y));
    col *= vec3<f32>(1.0 - clamp(1.0 - 2.0 * m.z * m.z, 0.0, 1.0)) * 7.0;
    return col;
}

fn animation(uv: vec2<f32>, t: f32) -> vec3<f32> {
    if (length(uv) > 1.0) {
        return vec3<f32>(1.0);
    }
    let o = vec3<f32>(0.0, 0.0, -5.0);
    let r = normalize(vec3<f32>(uv, 0.5));
    var col = vec3<f32>(0.0);
    var d = 0.0;
    for (var i: i32 = 0; i < 70; i++) {
        let m = map(o + d * r, t);
        if (m.w < 0.001) {
            col = map_color(m) * pow(2.0, -d);
            break;
        }
        d += m.w;
    }
    return col;
}

fn phase(uv: vec2<f32>) -> f32 {
    return REPSPERROUND * atan2(uv.y, uv.x) / 2.0 / PI + RINGS * length(uv);
}

fn background(uv: vec2<f32>) -> vec3<f32> {
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
    return animation(uv, phase(uv));
}

fn foreground(uv: vec2<f32>, t: f32) -> vec3<f32> {
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
    let k = mod_f(phase(uv) - t, 1.0);
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
            switch (I_MODE) {
                default: {
                    let fg = foreground(uv, t);
                    if (fg.x != 0.0) {
                        col += fg * background(uv);
                    }
                    break;
                }
                case 1: {
                    col += background(uv);
                    break;
                }
                case 2: {
                    col += foreground(uv, t);
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
                        col += mix(background(uv), foreground(uv, t), alpha);
                    } else if (blend_step == 1) {
                        col += mix(foreground(uv, t), foreground(uv, t) * background(uv), alpha);
                    } else if (blend_step == 2) {
                        col += mix(foreground(uv, t) * background(uv), animation(uv, t), alpha);
                    } else {
                        col += mix(animation(uv, t), background(uv), alpha);
                    }
                }
            }
        }
    }
    col = col / f32(AA * AA);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
