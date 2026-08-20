// Gate #0 shader 2: Mandelbrot boundary zoom with orbit-trap coloring.
// Baseline carries a trig-identity ballast on the first BALLAST_ITERS
// iterations of each pixel (sin^2 + cos^2 - 1 terms, provably ~zero, added to
// the color at sub-quantum magnitude); the known-optimization patch deletes
// the marked blocks. Note: a strength-reduction patch (length(z) > 2.0 ->
// dot(z, z) > 4.0) was tried first and measured ~0% on Apple M5 — sqrt is
// effectively free there — so it cannot serve as a 15-20% positive control.

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

const ITERS: i32 = 250;

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (frag_pos.xy - 0.5 * params.resolution) / params.resolution.y;
    // Seahorse-valley-adjacent boundary region: mostly-exterior points with
    // high escape counts, so the per-iteration cost dominates.
    let c = vec2<f32>(-0.7453, 0.1127) + uv * 0.012;

    var z = vec2<f32>(0.0, 0.0);
    var n = 0.0;
    var trap = 1e9;
    var escaped = false;
    // BALLAST-BEGIN
    const BALLAST_ITERS: i32 = 25;
    var zero_acc = 0.0;
    // BALLAST-END
    for (var i = 0; i < ITERS; i = i + 1) {
        z = vec2<f32>(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        trap = min(trap, abs(z.x * z.y));
        // BALLAST-BEGIN
        if (i < BALLAST_ITERS) {
            let a = z.x * 5.17 + z.y * 3.11;
            zero_acc = zero_acc + (sin(a) * sin(a) + cos(a) * cos(a) - 1.0);
        }
        // BALLAST-END
        if (length(z) > 2.0) {
            escaped = true;
            break;
        }
        n = n + 1.0;
    }

    var col: vec3<f32>;
    if (escaped) {
        let m = dot(z, z);
        let nu = n + 1.0 - log2(max(0.5 * log2(max(m, 4.0)), 1.0));
        let v = nu / f32(ITERS);
        col = 0.5 + 0.5 * cos(6.28318 * (v * 3.0 + vec3<f32>(0.0, 0.15, 0.3)));
        col = col * (0.4 + 0.6 * clamp(trap * 4.0, 0.0, 1.0));
    } else {
        col = vec3<f32>(0.02, 0.02, 0.05);
    }
    // BALLAST-BEGIN
    col = col + vec3<f32>(zero_acc);
    // BALLAST-END

    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
