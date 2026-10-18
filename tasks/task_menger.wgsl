// Original: Menger Sponge, by Inigo Quilez (2013); XR adaptation by Matthew Arcus.
// URL: https://www.shadertoy.com/view/4sX3Rn
// Article: http://www.iquilezles.org/www/articles/menger/menger.htm
// Source: tasks/sources/round2/arcus_menger_iq.glsl.
// License: MIT; exact original statement: source lines 1-3 (iq 2013);
// XR license: tasks/sources/round2/LICENSE-xrshader, lines 1-21 (Arcus 2021).
// Contamination flag: HIGH (iq classic 4sX3Rn).
// Port date: 2026-10-17. Faithful GLSL-to-WGSL port; no optimization.
// Port decision: activate canonical Shadertoy mainImage, source lines 135-170,
// despite its #if 0 wrapper; drop the mainVR XR adaptation, lines 172-179.
// Choose HW_PERFORMANCE != 0, AA = 2; both AA loops and in-loop camera kept.
// ZERO = min(nonnegative frame index,0) baked to 0; iMouse.xy = vec2(0,0),
// with no remaining mouse use after dropping mainVR. AA<=1 stays inactive.
// Port changes: house uniform/entry points; mutable p uses a copy; original
// signed mod uses floor-mod; matrix columns and vector splats are explicit.
// The dormant entry adds vec4 render output to vec3 col: take render(...).xyz.
// The original normalized light is initialized per invocation at entry.
// Final RGB is clamped to [0,1] to match framebuffer clamp; alpha is 1.0.
// Vertical orientation mirrored vs the GL original (WebGPU fragment coordinates); consistent across reference and submissions. No Y flip.
// The MIT License
// Copyright © 2013 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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

const AA: i32 = 2;
const ZERO: i32 = 0;

fn mod_v3(x: vec3<f32>, y: vec3<f32>) -> vec3<f32> {
    return x - y * floor(x / y);
}

fn maxcomp(p: vec3<f32>) -> f32 { return max(p.x, max(p.y, p.z)); }

fn sdBox(p: vec3<f32>, b: vec3<f32>) -> f32 {
    let di = abs(p) - b;
    let mc = maxcomp(di);
    return min(mc, length(max(di, vec3<f32>(0.0))));
}

const ma: mat3x3<f32> = mat3x3<f32>(vec3<f32>(0.60, 0.00, 0.80),
                                  vec3<f32>(0.00, 1.00, 0.00),
                                  vec3<f32>(-0.80, 0.00, 0.60));

fn map(p_in: vec3<f32>) -> vec4<f32> {
    var p = p_in;
    var d = sdBox(p, vec3<f32>(1.0));
    var res = vec4<f32>(d, 1.0, 0.0, 0.0);
    let ani = smoothstep(-0.2, 0.2, -cos(0.5 * params.time));
    let off = 1.5 * sin(0.01 * params.time);
    var s = 1.0;
    for (var m: i32 = 0; m < 4; m++) {
        p = mix(p, ma * (p + vec3<f32>(off)), ani);
        let a = mod_v3(p * s, vec3<f32>(2.0)) - vec3<f32>(1.0);
        s *= 3.0;
        let r = abs(vec3<f32>(1.0) - 3.0 * abs(a));
        let da = max(r.x, r.y);
        let db = max(r.y, r.z);
        let dc = max(r.z, r.x);
        let c = (min(da, min(db, dc)) - 1.0) / s;
        if (c > d) {
            d = c;
            res = vec4<f32>(d, min(res.y, 0.2 * da * db * dc), (1.0 + f32(m)) / 4.0, 0.0);
        }
    }
    return res;
}

fn intersect(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    var t = 0.0;
    var res = vec4<f32>(-1.0);
    var h = vec4<f32>(1.0);
    for (var i: i32 = 0; i < 64; i++) {
        if (h.x < 0.002 || t > 10.0) { break; }
        h = map(ro + rd * t);
        res = vec4<f32>(t, h.yzw);
        t += h.x;
    }
    if (t > 10.0) { res = vec4<f32>(-1.0); }
    return res;
}

fn softshadow(ro: vec3<f32>, rd: vec3<f32>, mint: f32, k: f32) -> f32 {
    var res = 1.0;
    var t = mint;
    var h = 1.0;
    for (var i: i32 = 0; i < 32; i++) {
        h = map(ro + rd * t).x;
        res = min(res, k * h / t);
        t += clamp(h, 0.005, 0.1);
    }
    return clamp(res, 0.0, 1.0);
}

fn calcNormal(pos: vec3<f32>) -> vec3<f32> {
    let eps = vec3<f32>(0.001, 0.0, 0.0);
    var nor: vec3<f32>;
    nor.x = map(pos + eps.xyy).x - map(pos - eps.xyy).x;
    nor.y = map(pos + eps.yxy).x - map(pos - eps.yxy).x;
    nor.z = map(pos + eps.yyx).x - map(pos - eps.yyx).x;
    return normalize(nor);
}

var<private> light: vec3<f32>;

fn render(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    var col = vec4<f32>(mix(vec3<f32>(0.3, 0.2, 0.1) * 0.5, vec3<f32>(0.7, 0.9, 1.0), 0.5 + 0.5 * rd.y), 0.0);
    let tmat = intersect(ro, rd);
    if (tmat.x > 0.0) {
        let pos = ro + tmat.x * rd;
        let nor = calcNormal(pos);
        let occ = tmat.y;
        let sha = softshadow(pos, light, 0.01, 64.0);
        let dif = max(0.1 + 0.9 * dot(nor, light), 0.0);
        let sky = 0.5 + 0.5 * nor.y;
        let bac = max(0.4 + 0.6 * dot(nor, vec3<f32>(-light.x, light.y, -light.z)), 0.0);
        var lin = vec3<f32>(0.0);
        lin += 1.00 * dif * vec3<f32>(1.10, 0.85, 0.60) * sha;
        lin += 0.50 * sky * vec3<f32>(0.10, 0.20, 0.40) * occ;
        lin += 0.10 * bac * vec3<f32>(1.00) * (0.5 + 0.5 * occ);
        lin += 0.25 * occ * vec3<f32>(0.15, 0.17, 0.20);
        let matcol = vec3<f32>(0.5 + 0.5 * cos(0.0 + 2.0 * tmat.z),
                               0.5 + 0.5 * cos(1.0 + 2.0 * tmat.z),
                               0.5 + 0.5 * cos(2.0 + 2.0 * tmat.z));
        col = vec4<f32>(matcol * lin, 1.0);
    }
    return vec4<f32>(pow(col.xyz, vec3<f32>(0.4545)), col.w);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    light = normalize(vec3<f32>(1.0, 0.9, 0.3));
    let ro = 1.1 * vec3<f32>(2.5 * sin(0.25 * params.time), 1.0 + 1.0 * cos(params.time * 0.13), 2.5 * cos(0.25 * params.time));
    var col = vec3<f32>(0.0);
    for (var m: i32 = ZERO; m < AA; m++) {
        for (var n: i32 = ZERO; n < AA; n++) {
            let o = vec2<f32>(f32(m), f32(n)) / f32(AA) - vec2<f32>(0.5);
            let p = (2.0 * (frag_pos.xy + o) - params.resolution) / params.resolution.y;
            let ww = normalize(vec3<f32>(0.0) - ro);
            let uu = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), ww));
            let vv = normalize(cross(ww, uu));
            let rd = normalize(p.x * uu + p.y * vv + 2.5 * ww);
            col += render(ro, rd).xyz;
        }
    }
    col /= f32(AA * AA);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
