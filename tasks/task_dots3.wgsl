// Original: Dots3, by keijiro (ShaderSketches / KodeLife).
// URL: https://github.com/keijiro/ShaderSketches
// Source: tasks/sources/Dots3.glsl.
// License: Unlicense, stated under "License" in tasks/sources/keijiro-README.md,
// line 21 (Code: Unlicense).
// Port date: 2026-10-17. Faithful GLSL-to-WGSL port; no optimization.
// Port changes: house uniform/entry points; GLSL float/vector types and literals;
// two-argument atan becomes atan2; mod_f preserves floor-mod for negative rows;
// scalar vector offsets use explicit splats. No mode baking or identifier renames.
// Final RGB clamp reproduces the original implicit framebuffer clamp; alpha 1.
// Vertical orientation mirrored vs the GL original (WebGPU fragment coordinates); consistent across reference and submissions.

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

fn mod_f(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn swirl(coord: vec2<f32>) -> f32 {
    let l = length(coord) / params.resolution.x;
    let phi = atan2(coord.y, coord.x + 1e-6);
    return sin(l * 21.0 + phi * 5.0 - params.time * 4.0) * 0.5 + 0.5;
}

fn halftone(coord: vec2<f32>, size: f32, offs: vec2<f32>) -> f32 {
    let uv = coord / size;
    let ip = floor(uv) + offs; // column, row
    let odd = vec2<f32>(0.5 * mod_f(ip.y, 2.0), 0.0); // odd line offset
    let cp = floor(uv - odd + offs) + odd; // dot center
    let d = length(uv - cp - vec2<f32>(0.5)) * size; // distance
    let r = swirl(cp * size) * size * 0.6; // dot radius
    return max(0.0, d - r);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = frag_pos.xy - params.resolution * 0.5;
    let size = params.resolution.x / (30.0 + sin(params.time * 0.5) * 20.0);
    let k = size / 4.0;

    var d = halftone(coord, size, vec2<f32>(-0.5, -1.0));
    d = smin(d, halftone(coord, size, vec2<f32>(0.5, -1.0)), k);
    d = smin(d, halftone(coord, size, vec2<f32>(-1.0, 0.0)), k);
    d = smin(d, halftone(coord, size, vec2<f32>(0.0, 0.0)), k);
    d = smin(d, halftone(coord, size, vec2<f32>(1.0, 0.0)), k);
    d = smin(d, halftone(coord, size, vec2<f32>(-0.5, 1.0)), k);
    d = smin(d, halftone(coord, size, vec2<f32>(0.5, 1.0)), k);

    let col = vec3<f32>(d, d, d);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
