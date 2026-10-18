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

const PI: f32 = acos(-1.0);
const H: vec2<f32> = vec2<f32>(127.1, 269.5);
const HALF: vec2<f32> = vec2<f32>(0.5);
const BIAS: vec2<f32> = vec2<f32>(0.1);
const OFFSETS: array<vec2<f32>, 12> = array<vec2<f32>, 12>(
    HALF + 0.5 * sin(3.0 * fract(sin(H * 0.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 1.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 2.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 3.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 4.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 5.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 6.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 7.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 8.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 9.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 10.0) * 18.5453) - BIAS),
    HALF + 0.5 * sin(3.0 * fract(sin(H * 11.0) * 18.5453) - BIAS)
);

fn mod_f(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn voronoi(x: f32, y: f32) -> f32 {
    let n = floor(x) + floor(y);
    let f = fract(vec2<f32>(x, y));
    var offsets: array<vec2<f32>, 5>;
    for (var k: i32 = 0; k < 5; k++) {
        offsets[k] = OFFSETS[u32(mod_f(n + f32(k - 2), 12.0))];
    }
    var d = 8.0;
    for (var j: i32 = -1; j <= 1; j++) {
        for (var i: i32 = -1; i <= 1; i++) {
            let r = vec2<f32>(f32(i), f32(j)) - f + offsets[i + j + 2];
            d = min(d, dot(r, r));
        }
    }
    return sqrt(d);
}

fn animation(uv: vec2<f32>, t: f32) -> vec3<f32> {
    let r = normalize(vec3<f32>(uv, 0.5));
    let u = atan2(r.y, r.x) / PI;
    let angular_x = mod_f((u - t / 12.0) * 12.0, 12.0);
    let initial_x = mod_f((-t / 12.0) * 12.0, 12.0);
    let initial_q = voronoi(initial_x, -20.0 + t);
    var d = 1.0 - initial_q / 3.0;
    let radial_speed = length(r.xy);
    for (var i: i32 = 1; i < 70; i++) {
        let v = (-5.0 + d * r.z) * 4.0;
        let q = voronoi(angular_x, v + t);
        let distance = 1.0 - d * radial_speed - q / 3.0;
        if (distance < 0.001) {
            var col = vec3<f32>(0.6) + 0.4 * cos(vec3<f32>(u * PI) + vec3<f32>(0.0, 0.5, 1.0) + vec3<f32>(v));
            col *= vec3<f32>(1.0 - clamp(1.0 - 2.0 * q * q, 0.0, 1.0)) * 7.0;
            return col * exp2(-d);
        }
        d += distance;
    }
    return vec3<f32>(0.0);
}

fn phase(uv: vec2<f32>) -> f32 {
    return 16.0 * atan2(uv.y, uv.x) / 2.0 / PI + 50.0 * length(uv);
}

fn foreground(uv: vec2<f32>, t: f32) -> bool {
    let maskrad = 139.2 / 145.0;
    if (length(uv / maskrad - vec2<f32>(0.72)) < 0.01) {
        return false;
    }
    if (length(uv / maskrad - vec2<f32>(0.76, 0.69)) < 0.02) {
        return false;
    }
    if (length(uv) > maskrad) {
        return true;
    }
    if (length(uv) < 3.0 / 145.0) {
        return true;
    }
    return mod_f(phase(uv) - t, 1.0) <= 0.3;
}

fn background_kind(uv: vec2<f32>) -> i32 {
    if (length(uv - vec2<f32>(0.72)) < 0.01) {
        return 0;
    }
    if (length(uv - vec2<f32>(0.76, 0.69)) < 0.02) {
        return 0;
    }
    if (abs(max(abs(uv.x), abs(uv.y)) - 1.0) < 0.002) {
        return 0;
    }
    let radius = length(uv);
    if (radius < 3.0 / 145.0 || radius > 1.0) {
        return 1;
    }
    return 2;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    var white_count = 0.0;
    var uv_sum = vec2<f32>(0.0);
    var count = 0.0;
    let t = mod_f(params.time, 1.0);
    for (var x: i32 = 0; x < 4; x++) {
        for (var y: i32 = 0; y < 4; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) / 4.0;
            let uv = ((frag_pos.xy + offset) * 2.0 - params.resolution) / params.resolution.y;
            if (foreground(uv, t)) {
                let kind = background_kind(uv);
                if (kind == 1) {
                    white_count += 1.0;
                } else if (kind == 2) {
                    uv_sum += uv;
                    count += 1.0;
                }
            }
        }
    }
    var col = vec3<f32>(white_count);
    if (count > 0.0) {
        let uv = uv_sum / count;
        col += count * animation(uv, phase(uv));
    }
    return vec4<f32>(clamp(col / 16.0, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}