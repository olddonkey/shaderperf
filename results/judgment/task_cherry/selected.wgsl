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

fn rand(uv: vec2<f32>) -> f32 {
    return fract(sin(dot(uv, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    var scale = params.resolution.y / 5.0;
    var p = frag_pos.xy / scale;
    let t = params.time * 0.6 + rand(floor(p)) * 200.0;
    let s2 = 1.6 + 0.6 * sin(t);
    p = (fract(p) - vec2<f32>(0.5)) * s2;
    scale /= s2;

    let phi = 3.1415926 * 0.1 + t;
    let v = vec2<f32>(cos(phi), sin(phi));
    let q = vec2<f32>(
        dot(p, v),
        abs(dot(p, vec2<f32>(-v.y, v.x)))
    );

    let q1 = vec2<f32>(
        0.309016994 * q.x + 0.951056516 * q.y,
        -0.951056516 * q.x + 0.309016994 * q.y
    );
    let q2 = vec2<f32>(
        -0.809016994 * q.x + 0.587785252 * q.y,
        -0.587785252 * q.x - 0.809016994 * q.y
    );
    let a = select(q, q1, q1.x > q.x);
    let b = select(a, q2, q2.x > a.x);

    let delta = vec2<f32>(b.x - 0.27, abs(b.y) + 0.15);
    let d1 = length(delta);
    let d2 = max(0.0, 0.540302306 * (b.x - 0.4) - 0.841470985 * abs(b.y));
    let coverage = 1.0 - max(d1 - 0.29, d2) * scale;
    let c = max(vec3<f32>(1.0, 0.7, 0.7) + vec3<f32>(p.y), vec3<f32>(0.0)) * coverage;
    return vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}