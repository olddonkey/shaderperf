// Gate #0 shader 1: raymarched SDF scene (spheres + box + wavy ground).
// Baseline deliberately lacks early loop termination: the march loop always
// runs MAX_STEPS iterations, with the body guarded so post-hit / escaped rays
// do no state updates. The known-optimization patch inserts the early exit at
// the marked site inside the loop. Output is bit-identical because sky color
// depends only on the ray direction and hit-lane state freezes at the hit.

struct Params {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
};
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

const MAX_STEPS: i32 = 56;
const TMAX: f32 = 24.0;
const EPS: f32 = 0.0015;

fn sd_sphere(p: vec3<f32>, r: f32) -> f32 { return length(p) - r; }

fn sd_box(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn map(p: vec3<f32>) -> f32 {
    let disp = 0.04 * sin(4.0 * p.x) * sin(4.0 * p.y + params.time) * sin(4.0 * p.z);
    let ground = p.y + 1.0 + 0.08 * sin(0.7 * p.x) * sin(0.9 * p.z);
    let s1 = sd_sphere(p - vec3<f32>(0.0, 0.2, 0.0), 1.0) + disp;
    let b1 = sd_box(p - vec3<f32>(2.1, -0.4, 1.2), vec3<f32>(0.6, 0.6, 0.6)) + disp;
    let s2 = sd_sphere(p - vec3<f32>(-2.0, 0.4, -0.6), 0.8) + disp;
    return min(min(ground, s1), min(b1, s2));
}

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.001;
    let k1 = vec3<f32>(1.0, -1.0, -1.0);
    let k2 = vec3<f32>(-1.0, -1.0, 1.0);
    let k3 = vec3<f32>(-1.0, 1.0, -1.0);
    let k4 = vec3<f32>(1.0, 1.0, 1.0);
    return normalize(
        k1 * map(p + k1 * e) + k2 * map(p + k2 * e) +
        k3 * map(p + k3 * e) + k4 * map(p + k4 * e));
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (frag_pos.xy - 0.5 * params.resolution) / params.resolution.y;

    let ro = vec3<f32>(0.0, 1.3, -5.0);
    let fwd = normalize(vec3<f32>(0.0, -0.22, 1.0));
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), fwd));
    let up = cross(fwd, right);
    let rd = normalize(fwd * 1.6 + right * uv.x - up * uv.y);

    var t = 0.0;
    var hit = false;
    for (var i = 0; i < MAX_STEPS; i = i + 1) {
        if (hit || t > TMAX) { break; }
        if (!hit && t <= TMAX) {
            let d = map(ro + rd * t);
            if (d < EPS) {
                hit = true;
            } else {
                t = t + min(d, 2.0);
            }
        }
    }

    var col: vec3<f32>;
    if (hit) {
        let pos = ro + rd * t;
        let n = calc_normal(pos);
        let l = normalize(vec3<f32>(0.6, 0.7, -0.4));
        let diff = clamp(dot(n, l), 0.0, 1.0);
        let amb = 0.15 + 0.1 * n.y;
        let base = mix(vec3<f32>(0.8, 0.5, 0.3), vec3<f32>(0.3, 0.5, 0.8),
                       clamp(0.5 + 0.5 * sin(pos.x * 2.0), 0.0, 1.0));
        col = base * (amb + 0.9 * diff);
        // distance fog
        col = mix(col, vec3<f32>(0.65, 0.75, 0.9), 1.0 - exp(-0.02 * t * t));
    } else {
        col = mix(vec3<f32>(0.65, 0.75, 0.9), vec3<f32>(0.2, 0.4, 0.8),
                  clamp(rd.y * 2.0 + 0.2, 0.0, 1.0));
    }

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
