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

// REFERENCE SOLUTION — never shown to the agent
// Second attempt: algebraic work reduction; all seven candidates are evaluated.
// Transformations (the provenance above describes the immutable starting port):
// 1. Share coord/size, floor(uv), row offsets, and time*4 across the seven dots:
//    these values are independent of the candidate, with the same grid centers.
// 2. Replace atan2 followed by the fifth angular harmonic with z^5, where
//    z = normalize(vec2(coord.x + 1e-6, coord.y)); for phi = atan2(y,x),
//    z^5 = (cos(5*phi), sin(5*phi)), an identity, not a series approximation.
//    The shifted vector is nonzero on the half-integer grid at pixel widths >= 1.
//    Share u = z.x^2 and q = u*(16*u-12): cos(5*phi)=z.x*(q-8*u+5),
//    sin(5*phi)=z.y*(q+1), the exact fifth-harmonic polynomials on the unit circle.
// 3. Use sin(A+5*phi) = sin(A)*cos(5*phi) + cos(A)*sin(5*phi), with
//    A = length(coord)/resolution.x*21 - time*4; the radial length remains
//    unshifted and the original Cartesian atan2 epsilon remains exactly 1e-6.
// 4. Work in grid-distance units until the final multiply by positive size:
//    max and smin are positively homogeneous, and (size/4)/size = 0.25,
//    so the geometry, smooth-min width, and final pixel-distance clamp agree.
// No candidate pruning, grid-count changes, or new approximation/epsilon.
// Algebraic reassociation can change f32 roundoff; GPU equivalence/timing
// remain maintainer gates, not claims established by this source.

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

fn smin_grid(a: f32, b: f32) -> f32 {
    let h = clamp(0.5 + 2.0 * (b - a), 0.0, 1.0);
    return mix(b, a, h) - 0.25 * h * (1.0 - h);
}

fn swirl(coord: vec2<f32>, time_phase: f32) -> f32 {
    let l = length(coord) / params.resolution.x;
    let z = normalize(vec2<f32>(coord.x + 1e-6, coord.y));
    let u = z.x * z.x;
    let q = u * (16.0 * u - 12.0);
    let z5 = vec2<f32>(z.x * (q - 8.0 * u + 5.0), z.y * (q + 1.0));
    let radial_phase = l * 21.0 - time_phase;
    let wave = sin(radial_phase) * z5.x + cos(radial_phase) * z5.y;
    return wave * 0.5 + 0.5;
}

fn dot_center(uv: vec2<f32>, offs: vec2<f32>, odd: vec2<f32>) -> vec2<f32> {
    return floor(uv - odd + offs) + odd;
}

fn halftone_grid(uv: vec2<f32>, size: f32, cp: vec2<f32>, time_phase: f32) -> f32 {
    let d = length(uv - cp - vec2<f32>(0.5));
    let r = swirl(cp * size, time_phase) * 0.6;
    return max(0.0, d - r);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = frag_pos.xy - params.resolution * 0.5;
    let size = params.resolution.x / (30.0 + sin(params.time * 0.5) * 20.0);

    let uv = coord / size;
    let base = floor(uv);
    let odd_below = vec2<f32>(0.5 * mod_f(base.y - 1.0, 2.0), 0.0);
    let odd_middle = vec2<f32>(0.5 * mod_f(base.y, 2.0), 0.0);
    let odd_above = vec2<f32>(0.5 * mod_f(base.y + 1.0, 2.0), 0.0);
    let time_phase = params.time * 4.0;

    var d = halftone_grid(uv, size, dot_center(uv, vec2<f32>(-0.5, -1.0), odd_below), time_phase);
    d = smin_grid(d, halftone_grid(uv, size, dot_center(uv, vec2<f32>(0.5, -1.0), odd_below), time_phase));
    d = smin_grid(d, halftone_grid(uv, size, dot_center(uv, vec2<f32>(-1.0, 0.0), odd_middle), time_phase));
    d = smin_grid(d, halftone_grid(uv, size, dot_center(uv, vec2<f32>(0.0, 0.0), odd_middle), time_phase));
    d = smin_grid(d, halftone_grid(uv, size, dot_center(uv, vec2<f32>(1.0, 0.0), odd_middle), time_phase));
    d = smin_grid(d, halftone_grid(uv, size, dot_center(uv, vec2<f32>(-0.5, 1.0), odd_above), time_phase));
    d = smin_grid(d, halftone_grid(uv, size, dot_center(uv, vec2<f32>(0.5, 1.0), odd_above), time_phase));

    let col = vec3<f32>(d * size);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
