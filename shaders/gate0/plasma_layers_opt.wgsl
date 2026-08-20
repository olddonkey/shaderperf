// Gate #0 shader 5: layered trigonometric plasma (pure-ALU load, no hashing).
// Baseline carries a BALLAST block on the first BALLAST_LAYERS of the
// LAYERS-iteration accumulation loop: sin^2 + cos^2 - 1 terms, provably zero
// up to float error, accumulated and added to the color. The
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

const LAYERS: i32 = 24;

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (frag_pos.xy - 0.5 * params.resolution) / params.resolution.y;
    let p = uv * 6.0;

    var v = 0.0;
    for (var l = 0; l < LAYERS; l = l + 1) {
        let fl = f32(l);
        let dir = vec2<f32>(sin(fl * 2.4 + 1.0), cos(fl * 1.7 + 2.0));
        let freq = 1.5 + 0.35 * fl;
        let phase = params.time * (0.3 + 0.06 * fl);
        v = v + sin(dot(p, dir) * freq + phase) / (1.0 + 0.15 * fl);
    }

    var col = 0.5 + 0.5 * cos(vec3<f32>(v * 1.2, v * 1.2 + 2.1, v * 1.2 + 4.2));
    col = col * (0.6 + 0.4 * sin(v * 3.0));

    return vec4<f32>(clamp(abs(col), vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
