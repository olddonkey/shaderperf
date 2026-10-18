// Original: codebasealpha, by essenbee (Stu Bonham).
// Copyright (c) 2019 Stu Bonham. MIT License.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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

fn reflect_fold(p: vec2<f32>, n: vec2<f32>) -> vec2<f32> {
    return p - (2.0 * min(dot(p, n), 0.0)) * n;
}

fn distance_squared(uv: vec2<f32>, c: f32, s: f32) -> f32 {
    var p = vec2<f32>(uv.x, abs(uv.y));
    p = reflect_fold(p, vec2<f32>(0.9945218953682733, 0.1045284632676535));
    p = reflect_fold(p, vec2<f32>(0.7431448254773942, -0.6691306063588582));
    p = reflect_fold(p, vec2<f32>(0.4067366430758002, -0.9135454576426009));
    p = reflect_fold(p, vec2<f32>(0.2079116908177593, -0.9781476007338057));

    p = vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
    let shifted = p + vec2<f32>(0.25);
    let local = shifted - vec2<f32>(0.5) * floor(shifted * 2.0)
        - vec2<f32>(0.25);
    let box = abs(local - vec2<f32>(0.1));
    let b = max(box.x, box.y);
    return min(dot(local, local), b * b);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    var uv = frag_pos.xy / params.resolution - vec2<f32>(0.5);
    uv.x *= params.resolution.x / params.resolution.y;

    let c = cos(params.time);
    let s = sin(params.time);
    let q2 = distance_squared(uv, c, s);

    if (q2 > 0.0064 && q2 < 0.0144) {
        return vec4<f32>(1.0);
    }

    let rotated = vec2<f32>(
        c * uv.x + s * uv.y,
        -s * uv.x + c * uv.y
    );
    let colored = q2 < 0.0001
        || (q2 >= 0.04 && q2 < 0.0441)
        || (q2 >= 0.09 && q2 < 0.0961);

    if (colored) {
        let d = sqrt(q2) - 0.1;
        let nd = abs(d / 0.1);
        let base = select(
            vec3<f32>(0.3, 1.0, 0.15),
            vec3<f32>(1.0, 0.3, 0.15),
            d < 0.0
        );
        let exponent = vec3<f32>(abs(rotated), length(rotated));
        let col = clamp(base / nd, vec3<f32>(0.0), vec3<f32>(1.0));
        return vec4<f32>(
            clamp(pow(col, exponent), vec3<f32>(0.0), vec3<f32>(1.0)),
            1.0
        );
    }

    return vec4<f32>(
        select(vec2<f32>(0.0), vec2<f32>(1.0), rotated == vec2<f32>(0.0)),
        0.0,
        1.0
    );
}