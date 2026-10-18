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

fn mod_f(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn gear(angle: f32, uv: vec2<f32>, wheels: f32, prev: ptr<function, vec4<f32>>) -> f32 {
    let radius = wheels / 50.0;
    let offset = vec2<f32>(cos(angle), sin(angle)) *
        ((*prev).z / 50.0 + radius + 0.06);
    let state = vec4<f32>(
        (*prev).xy + offset,
        wheels,
        -((*prev).w + angle * (*prev).z) - PI - (PI + angle) * wheels
    );
    *prev = state;
    let pos = uv - state.xy;
    let r2 = dot(pos, pos);
    let inner = radius - 0.06;
    let outer = radius + 0.052;
    if (r2 < inner * inner || r2 > outer * outer) {
        return 0.0;
    }
    let r = sqrt(r2);
    let a = atan2(pos.y, pos.x);
    let f = smoothstep(-0.5, 0.5, cos(a * wheels + state.w)) * 0.05 + radius;
    return (1.0 - smoothstep(f, f + 0.002, r)) * step(inner, r);
}

fn animation(uv_in: vec2<f32>, t: f32) -> vec3<f32> {
    if (length(uv_in) > 1.0) {
        return vec3<f32>(1.0);
    }
    var uv = uv_in / 1.4;
    var c = vec3<f32>(1.7);
    var p = vec3<f32>(0.0);
    let angle = 4.3 + PI;
    uv *= mat2x2<f32>(cos(angle), sin(angle), -sin(angle), cos(angle));
    for (var i = 0; i < 10; i++) {
        if (dot(uv, uv) > 1.69) {
            break;
        }
        var prev = vec4<f32>(0.0, 0.0, 15.0, t * 2.0 * PI);
        var g = 0.0;
        g = max(g, gear(0.0, uv, 10.0, &prev));
        g = max(g, gear(1.4, uv, 6.0, &prev));
        g = max(g, gear(3.2, uv, 6.0, &prev));
        g = max(g, gear(1.5, uv, 7.0, &prev));
        g = max(g, gear(3.4, uv, 8.0, &prev));
        g = max(g, gear(4.2, uv, 9.0, &prev));
        g = max(g, gear(2.3, uv, 6.0, &prev));
        g = max(g, gear(4.0, uv, 7.0, &prev));
        g = max(g, gear(4.3, uv, 5.0, &prev));
        g = max(g, gear(5.7, uv, 10.0, &prev));
        g = max(g, gear(4.8, uv, 6.0, &prev));
        g = max(g, gear(6.0, uv, 8.0, &prev));
        g = max(g, gear(1.2, uv, 9.0, &prev));
        g = max(g, gear(-0.7, uv, 5.0, &prev));
        g = max(g, gear(-0.2, uv, 6.0, &prev));
        g = max(g, gear(0.76, uv, 8.0, &prev));
        p = max(p, c * g);
        c *= vec3<f32>(0.75, 0.5, 0.7);
        if (all(p >= c)) {
            break;
        }
        uv *= 1.2;
        let a = f32(i) + 4.5;
        uv *= mat2x2<f32>(cos(a), sin(a), -sin(a), cos(a));
    }
    return pow(p, vec3<f32>(2.2));
}

fn phase(uv: vec2<f32>) -> f32 {
    return 16.0 * atan2(uv.y, uv.x) / 2.0 / PI + 50.0 * length(uv);
}

fn composite(uv: vec2<f32>, t: f32) -> vec3<f32> {
    let maskrad = 139.2 / 145.0;
    let r = length(uv);
    let fg_uv = uv / maskrad;
    if (length(fg_uv - vec2<f32>(0.72)) < 0.01 ||
        length(fg_uv - vec2<f32>(0.76, 0.69)) < 0.02) {
        return vec3<f32>(0.0);
    }
    var ph = 0.0;
    if (r <= maskrad && r >= 3.0 / 145.0) {
        ph = phase(uv);
        if (mod_f(ph - t, 1.0) > 0.3) {
            return vec3<f32>(0.0);
        }
    }
    if (length(uv - vec2<f32>(0.72)) < 0.01 ||
        length(uv - vec2<f32>(0.76, 0.69)) < 0.02 ||
        abs(max(abs(uv.x), abs(uv.y)) - 1.0) < 0.002) {
        return vec3<f32>(0.0);
    }
    if (r < 3.0 / 145.0 || r > 1.0) {
        return vec3<f32>(1.0);
    }
    if (r > maskrad) {
        ph = phase(uv);
    }
    return animation(uv, ph);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    var col = vec3<f32>(0.0);
    let t = mod_f(params.time, 1.0);
    for (var x = 0; x < 2; x++) {
        for (var y = 0; y < 2; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * 0.5 + vec2<f32>(0.125);
            let uv = ((frag_pos.xy + offset) * 2.0 -
                params.resolution) / params.resolution.y;
            col += composite(uv, t);
        }
    }
    return vec4<f32>(clamp(col * 0.25, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}