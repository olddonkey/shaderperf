// Calibration-only temporal scene: animated directional and radial interference.

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

const WAVE_TERMS: i32 = 16;

fn interference(p: vec2<f32>, time_value: f32) -> f32 {
    var signal = 0.0;
    var weight_sum = 0.0;
    for (var i = 0; i < WAVE_TERMS; i = i + 1) {
        let fi = f32(i);
        let direction = vec2<f32>(cos(fi * 2.399963), sin(fi * 2.399963));
        let frequency = 4.2 + fi * 0.73;
        let speed = 1.15 + fi * 0.09;
        let weight = 1.0 / (1.0 + fi * 0.11);
        let center = 0.42 * vec2<f32>(cos(fi * 1.37), sin(fi * 1.91));
        let plane_wave = sin(dot(p, direction) * frequency + time_value * speed);
        let ring_wave = sin(length(p - center) * (frequency + 2.7) - time_value * speed * 1.63);
        signal = signal + weight * (plane_wave + ring_wave * 0.72);
        weight_sum = weight_sum + weight * 1.72;
    }
    return signal / weight_sum;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (frag_pos.xy - 0.5 * params.resolution) / params.resolution.y;
    let wave = interference(uv * 3.4, params.time);
    let displaced = interference(uv * 5.1 + vec2<f32>(wave * 0.18, -wave * 0.13), params.time * 1.37 + 0.4);
    let crest = smoothstep(0.13, 0.54, abs(displaced));
    let phase = wave * 2.8 + displaced * 1.9 + params.time * 0.42;
    var col = 0.5 + 0.5 * cos(phase + vec3<f32>(0.0, 2.05, 4.10));
    col = mix(col * 0.48, col, crest);
    col = col + vec3<f32>(0.10, 0.17, 0.25) * (1.0 - crest);

    // SAFE-BEGIN
    var ballast = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let angle = phase * 3.1 + f32(i) * 0.731;
        ballast = ballast + sin(angle) * sin(angle) + cos(angle) * cos(angle) - 1.0;
    }
    col = col + vec3<f32>(ballast * 0.0000001);
    // SAFE-END

    // POSTERIZE-SITE
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
