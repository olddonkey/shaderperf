// Original: codebasealpha, by essenbee (Stu Bonham).
// URL: https://github.com/essenbee (author; upstream shader URL absent from snapshot).
// Source: tasks/sources/codebasealpha.glsl.
// License: MIT, stated in tasks/sources/LICENSE (copyright 2019 Stu Bonham).
// Port date: 2026-10-17. Faithful GLSL-to-WGSL port; no optimization.
// Port changes: house uniform/entry points; PI/TAU macros become const;
// floor-mod uses mod_f/mod_v2; two-argument atan becomes atan2; scalar
// vector offsets and clamp bounds use explicit splats. GLSL inout parameters
// in pMod2/rotate become function pointers; mutable pos/col parameters use
// pos_in/col_in aliases. No mode baking; unused helpers/computations retained.
// Final RGB clamp reproduces the original implicit framebuffer clamp; alpha 1.
// Vertical orientation mirrored vs the GL original (WebGPU fragment coordinates); consistent across reference and submissions.
//
// MIT License
//
// Copyright (c) 2019 Stu Bonham
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

fn mod_f(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn mod_v2(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    return x - y * floor(x / y);
}

fn dCircle(pos: vec2<f32>, radius: f32) -> f32 {
    return length(pos) - radius;
}

fn dRoundBox(pos_in: vec2<f32>, radius: f32) -> f32 {
    var pos = pos_in;
    pos = pos * pos;
    pos = pos * pos;
    let d8 = dot(pos, pos);
    return pow(d8, 1.0 / 8.0) - radius;
}

fn dBox(pos_in: vec2<f32>, siz: vec2<f32>) -> f32 {
    var pos = pos_in;
    pos = abs(pos);
    let dx = pos.x - siz.x;
    let dy = pos.y - siz.y;
    return max(dx, dy);
}

fn toPolarCoords(rectCoords: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(length(rectCoords), atan2(rectCoords.y, rectCoords.x));
}

fn toRectCoords(polarCoords: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(polarCoords.x * cos(polarCoords.y), polarCoords.x * sin(polarCoords.y));
}

const PI: f32 = 3.141592654;
const TAU: f32 = 2.0 * PI;

// Repeat in two dimensions
fn pMod2(p: ptr<function, vec2<f32>>, size: vec2<f32>) -> vec2<f32> {
    let c = floor((*p + size * 0.5) / size);
    *p = mod_v2(*p + size * 0.5, size) - size * 0.5;
    return c;
}

fn dF2(pos_in: vec2<f32>) -> f32 {
    var pos = pos_in;
    var pp = toPolarCoords(pos);
    let a = TAU / 30.0;
    let np = pp.y / a;
    pp.y = mod_f(pp.y, a);
    let m2 = mod_f(np, 2.0);
    if (m2 > 1.0) {
        pp.y = a - pp.y;
    }

    pp.y += params.time;
    pos = toRectCoords(pp);

    pMod2(&pos, vec2<f32>(0.5));

    let d1 = dCircle(pos, 0.1);

    let d3 = dBox(pos - vec2<f32>(0.1), vec2<f32>(0.1, 0.1));
    let d = min(d1, d3);
    return d;
}

fn dF(pos_in: vec2<f32>) -> f32 {
    var pos = pos_in;
    //pos = abs(pos);
    var pPos = toPolarCoords(pos);
    pPos.y = mod_f(pPos.y, TAU / 7.0);
    pPos.y += params.time;

    // pPos.x *= 1.0 + pos.y;
    pos = toRectCoords(pPos);
    pMod2(&pos, vec2<f32>(0.5));

    let d1 = dCircle(pos, 0.1);

    let d3 = dBox(pos - vec2<f32>(0.1), vec2<f32>(0.1, 0.1));
    let d = min(d1, d3);
    return d;
}

fn rotate(pos: ptr<function, vec2<f32>>, angle: f32) {
    let c = cos(angle);
    let s = sin(angle);
    *pos = vec2<f32>(c * (*pos).x + s * (*pos).y, -s * (*pos).x + c * (*pos).y);
}

fn postProcess(col_in: vec3<f32>, pos_in: vec2<f32>) -> vec3<f32> {
    var col = col_in;
    var pos = pos_in;
    rotate(&pos, params.time);
    col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
    return pow(col, vec3<f32>(abs(pos.x), abs(pos.y), length(pos)));
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    var uv = frag_pos.xy / params.resolution - vec2<f32>(0.5);
    uv.x *= params.resolution.x / params.resolution.y;

    let d = dF2(uv);

    var col = vec3<f32>(0.0);

    let md = mod_f(d, 0.1);
    let nd = abs(d / 0.1);

    if (abs(md) < 0.01) {
        if (d < 0.0) {
            col = vec3<f32>(1.0, 0.3, 0.15) / nd;
        } else {
            col = vec3<f32>(0.3, 1.0, 0.15) / nd;
        }
    }

    if (abs(d) < 0.02) {
        col = vec3<f32>(1.0);
    }

    col = postProcess(col, uv);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
