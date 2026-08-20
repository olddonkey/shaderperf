// Smoke-test shader: procedural fbm + fake lighting. Self-written (no license
// concerns), LDR single-output fragment shader, no external randomness —
// matches the v0.1 task domain. Load is tuned to land in the hundreds of
// microseconds to low milliseconds on an Apple-Silicon-class GPU so timestamp
// resolution is not the dominant noise source.

struct Params {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: Params;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    // Fullscreen triangle.
    var out: VSOut;
    let x = f32(i32(vi & 1u) * 4 - 1);
    let y = f32(i32(vi >> 1u) * 4 - 1);
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

fn hash2(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn noise2(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i + vec2<f32>(0.0, 0.0));
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p_in: vec2<f32>) -> f32 {
    var p = p_in;
    var amp = 0.5;
    var sum = 0.0;
    let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    for (var i = 0; i < 8; i = i + 1) {
        sum = sum + amp * noise2(p);
        p = rot * p * 2.02;
        amp = amp * 0.5;
    }
    return sum;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_pos.xy / params.resolution;
    let p = (uv - 0.5) * vec2<f32>(params.resolution.x / params.resolution.y, 1.0) * 4.0;

    // Domain-warped fbm: three nested evaluations per pixel.
    let q = vec2<f32>(fbm(p + vec2<f32>(0.0, params.time * 0.1)),
                      fbm(p + vec2<f32>(5.2, 1.3)));
    let r = vec2<f32>(fbm(p + 4.0 * q + vec2<f32>(1.7, 9.2)),
                      fbm(p + 4.0 * q + vec2<f32>(8.3, 2.8)));
    let v = fbm(p + 4.0 * r);

    // Cheap normal-ish shading from finite differences to add ALU work.
    let e = 0.01;
    let vx = fbm(p + 4.0 * r + vec2<f32>(e, 0.0)) - v;
    let vy = fbm(p + 4.0 * r + vec2<f32>(0.0, e)) - v;
    let n = normalize(vec3<f32>(-vx / e, -vy / e, 1.0));
    let l = normalize(vec3<f32>(0.5, 0.7, 0.6));
    let diff = clamp(dot(n, l), 0.0, 1.0);

    let base = mix(vec3<f32>(0.1, 0.2, 0.35), vec3<f32>(0.9, 0.75, 0.5), v);
    let col = clamp(base * (0.25 + 0.75 * diff), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(col, 1.0);
}
