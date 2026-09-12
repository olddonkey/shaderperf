// Calibration-only smooth scene: gradient sky, sun disc, and one fbm cloud layer.

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

const CLOUD_OCTAVES: i32 = 6;

fn hash2(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
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

fn cloud_fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var amplitude = 0.52;
    var value = 0.0;
    var weight = 0.0;
    let rotation = mat2x2<f32>(0.80, 0.60, -0.60, 0.80);
    for (var i = 0; i < CLOUD_OCTAVES; i = i + 1) {
        value = value + amplitude * value_noise(p);
        weight = weight + amplitude;
        p = rotation * p * 2.03 + vec2<f32>(0.17, 0.31);
        amplitude = amplitude * 0.51;
    }
    return value / weight;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_pos.xy / params.resolution;
    let aspect = params.resolution.x / params.resolution.y;
    let horizon = smoothstep(0.0, 1.0, uv.y);
    let low_sky = vec3<f32>(0.84, 0.91, 0.98);
    let high_sky = vec3<f32>(0.09, 0.30, 0.67);
    var col = mix(low_sky, high_sky, pow(horizon, 0.72));

    let sun_position = vec2<f32>(0.73 * aspect, 0.28);
    let sky_position = vec2<f32>(uv.x * aspect, uv.y);
    let sun_distance = distance(sky_position, sun_position);
    let sun_disc = smoothstep(0.075, 0.055, sun_distance);
    let sun_glow = exp(-sun_distance * 15.0);
    col = mix(col, vec3<f32>(1.0, 0.82, 0.48), sun_disc);
    col = col + vec3<f32>(1.0, 0.62, 0.28) * sun_glow * 0.18;

    let drift = vec2<f32>(params.time * 0.035, params.time * 0.008);
    let cloud = cloud_fbm(vec2<f32>(uv.x * aspect, uv.y) * 4.2 + drift);
    let cloud_shape = smoothstep(0.49, 0.68, cloud + (0.57 - uv.y) * 0.10);
    let cloud_fade = smoothstep(0.12, 0.42, uv.y) * smoothstep(0.96, 0.57, uv.y);
    col = mix(col, vec3<f32>(0.96, 0.97, 0.99), cloud_shape * cloud_fade * 0.82);

    // SAFE-BEGIN
    var ballast = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let angle = cloud * 9.0 + f32(i) * 1.713;
        ballast = ballast + sin(angle) * sin(angle) + cos(angle) * cos(angle) - 1.0;
    }
    col = col + vec3<f32>(ballast * 0.0000001);
    // SAFE-END

    // POSTERIZE-SITE
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
