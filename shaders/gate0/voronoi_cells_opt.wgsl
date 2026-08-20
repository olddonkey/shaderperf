// Gate #0 shader 4: animated Voronoi cells with F1/F2 border shading over a
// 5x5 neighborhood. Baseline carries a BALLAST block: for the first
// BALLAST_CELLS cells it accumulates sin^2(a) + cos^2(a) - 1, which is
// provably zero (trig identity) up to ~1e-7 float error — far below the
// 8-bit output quantum — and adds the accumulator to the color. The
// known-optimization patch deletes the BALLAST blocks citing the identity;
// output is identical after quantization (verified by pixel readback).

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

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)),
                      dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_pos.xy / params.resolution;
    let p = uv * vec2<f32>(params.resolution.x / params.resolution.y, 1.0) * 9.0;
    let ip = floor(p);
    let fp = fract(p);

    var f1 = 8.0;
    var f2 = 8.0;
    var cell_id = vec2<f32>(0.0, 0.0);
    for (var j = -2; j <= 2; j = j + 1) {
        for (var i = -2; i <= 2; i = i + 1) {
            let g = vec2<f32>(f32(i), f32(j));
            let o = hash22(ip + g);
            let pt = g + 0.5 + 0.5 * sin(params.time + 6.2831 * o) - fp;
            let d = dot(pt, pt);
            if (d < f1) {
                f2 = f1;
                f1 = d;
                cell_id = ip + g;
            } else if (d < f2) {
                f2 = d;
            }
        }
    }

    let border = smoothstep(0.0, 0.15, f2 - f1);
    let tint = 0.5 + 0.5 * cos(6.28318 * (hash22(cell_id).x + vec3<f32>(0.0, 0.33, 0.67)));
    var col = tint * (0.3 + 0.7 * border) * (1.0 - 0.55 * sqrt(f1));

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
