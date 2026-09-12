// Calibration-only high-frequency scene: Voronoi borders mixed with value noise.

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

const DETAIL_OCTAVES: i32 = 7;

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

fn value_noise(p: vec2<f32>) -> f32 {
    let cell = floor(p);
    let local = fract(p);
    let blend = local * local * (3.0 - 2.0 * local);
    let a = hash2(cell);
    let b = hash2(cell + vec2<f32>(1.0, 0.0));
    let c = hash2(cell + vec2<f32>(0.0, 1.0));
    let d = hash2(cell + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, blend.x), mix(c, d, blend.x), blend.y);
}

fn detail_noise(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var amplitude = 0.52;
    var value = 0.0;
    var weight = 0.0;
    let rotation = mat2x2<f32>(0.82, 0.57, -0.57, 0.82);
    for (var i = 0; i < DETAIL_OCTAVES; i = i + 1) {
        value = value + amplitude * value_noise(p);
        weight = weight + amplitude;
        amplitude = amplitude * 0.53;
        p = rotation * p * 2.08 + vec2<f32>(0.37, 0.19);
    }
    return value / weight;
}

fn voronoi_gap(p: vec2<f32>, time_value: f32) -> f32 {
    let cell = floor(p);
    let local = fract(p);
    var nearest = 8.0;
    var second = 8.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let neighbor = vec2<f32>(f32(x), f32(y));
            let random_point = hash22(cell + neighbor);
            let animated = 0.5 + 0.34 * sin(time_value * 0.22 + 6.2831853 * random_point);
            let delta = neighbor + animated - local;
            let distance_squared = dot(delta, delta);
            if (distance_squared < nearest) {
                second = nearest;
                nearest = distance_squared;
            } else if (distance_squared < second) {
                second = distance_squared;
            }
        }
    }
    return sqrt(second) - sqrt(nearest);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_pos.xy / params.resolution;
    let aspect = params.resolution.x / params.resolution.y;
    let p = vec2<f32>(uv.x * aspect, uv.y) * 19.0;
    let gap = voronoi_gap(p, params.time);
    let borders = 1.0 - smoothstep(0.035, 0.105, gap);
    let fine = detail_noise(p * 0.78 + vec2<f32>(params.time * 0.025, 0.0));
    let grain = detail_noise(p * 2.35 - vec2<f32>(0.0, params.time * 0.018));
    let palette = 0.5 + 0.5 * cos(6.2831853 * (fine + vec3<f32>(0.00, 0.21, 0.43)));
    var col = mix(palette * (0.42 + 0.58 * grain), vec3<f32>(0.94, 0.96, 0.99), borders * 0.88);
    col = col * (0.82 + 0.18 * sin((uv.x - uv.y) * 240.0 + fine * 8.0));

    // SAFE-BEGIN
    var ballast = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let angle = gap * 31.0 + f32(i) * 1.191;
        ballast = ballast + sin(angle) * sin(angle) + cos(angle) * cos(angle) - 1.0;
    }
    col = col + vec3<f32>(ballast * 0.0000001);
    // SAFE-END

    // POSTERIZE-SITE
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
