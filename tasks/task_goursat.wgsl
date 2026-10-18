// Original: Goursat Quartic Surfaces, by Matthew Arcus (2021).
// URL: ./sources/round2/arcus_goursat.glsl (local snapshot;
// upstream permalink was not recorded with the supplied source).
// Source: tasks/sources/round2/arcus_goursat.glsl.
// License: MIT; exact in-file statement: source lines 3-5;
// full statement: tasks/sources/round2/LICENSE-xrshader, lines 1-21,
// copyright (c) 2021 Matthew Arcus.
// Port date: 2026-10-17. Faithful GLSL-to-WGSL port; no optimization.
// Port decisions: desktop mainImage path; omit mainVR and its XR view uniform.
// XR_MODE != AR_MODE is baked as XR_IS_AR = false, retaining the adaptive
// 2x2 AA loop for k >= 0.9 and one sample outside that region. iMouse.xy is
// vec2(0,0), with z/w also zero: automatic surface selection, AA enabled.
// Original #if 0 general quartic expansion stays inactive; original active
// p.r=0 expansion retained. Solver/debug/lighting structure is unchanged.
// Port changes: house uniform/entry points; dFdx/dFdy become dpdx/dpdy;
// out/inout become function pointers, with temporaries for swizzle outputs;
// swizzle assignments expand to component stores, parameters use local copies.
// Ternaries become select/if (division guarded for alpha==0); integer modulo
// remains typed integer remainder. Post-increment comparisons are sequenced.
// assert renamed debug_assert; local params renamed surface_params.
// Formerly uninitialized locals use WGSL zero initialization; normalized light
// initialized at entry, alert remains per invocation and persists across AA.
// Final RGB is clamped to [0,1] to match framebuffer clamp; alpha is 1.0
// (the original coverage alpha is discarded after its original gamma step).
// Vertical orientation mirrored vs the GL original (WebGPU fragment coordinates); consistent across reference and submissions. No Y flip.

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

const dorotate: bool = true;
var<private> light: vec3<f32>;
const ambient: f32 = 0.4;
const diffuse: f32 = 0.6;
const specular: f32 = 0.8;
const specularpow: f32 = 10.0;
const specularcolor: vec3<f32> = vec3<f32>(1.0);
var<private> alert: bool = false;
const PI: f32 = 3.141592654;
const MOUSE: vec4<f32> = vec4<f32>(0.0);
const XR_IS_AR: bool = false;

fn debug_assert(t: bool) {
    if (!t) { alert = true; }
}

fn rotate(p: vec2<f32>, t: f32) -> vec2<f32> {
    return p * cos(t) + vec2<f32>(p.y, -p.x) * sin(t);
}

fn sgn(x: f32) -> f32 {
    return select(1.0, -1.0, x < 0.0); // Return 1 for x == 0.
}

fn evalquadratic(x: f32, A: f32, B: f32, C: f32) -> f32 {
    return (A * x + B) * x + C;
}

fn evalcubic(x: f32, A: f32, B: f32, C: f32, D: f32) -> f32 {
    return ((A * x + B) * x + C) * x + D;
}

// Quadratic solver from Kahan.
fn quadratic(A: f32, B: f32, C: f32, res: ptr<function, vec2<f32>>) -> i32 {
    let b = -0.5 * B;
    let b2 = b * b;
    let q = b2 - A * C;
    if (q < 0.0) { return 0; }
    let r = b + sgn(b) * sqrt(q);
    if (r == 0.0) {
        (*res)[0] = C / A;
        (*res)[1] = -(*res)[0];
    } else {
        (*res)[0] = C / r;
        (*res)[1] = r / A;
    }
    return 2;
}

// Numerical Recipes algorithm for solving cubic equation.
fn cubic(a_in: f32, b_in: f32, c_in: f32, d: f32, res: ptr<function, vec3<f32>>) -> i32 {
    var a = a_in;
    var b = b_in;
    var c = c_in;
    if (a == 0.0) {
        var res_xy = (*res).xy;
        let count = quadratic(b, c, d, &res_xy);
        (*res).x = res_xy.x;
        (*res).y = res_xy.y;
        return count;
    }
    if (d == 0.0) {
        (*res).x = 0.0;
        var res_yz = (*res).yz;
        let count = quadratic(a, b, c, &res_yz);
        (*res).y = res_yz.x;
        (*res).z = res_yz.y;
        return 1 + count;
    }
    let tmp = a;
    a = b / tmp;
    b = c / tmp;
    c = d / tmp;
    // Solve x^3 + ax^2 + bx + c = 0.
    let Q = (a * a - 3.0 * b) / 9.0;
    let R = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0;
    let R2 = R * R;
    let Q3 = Q * Q * Q;
    if (R2 < Q3) {
        let X = clamp(R / sqrt(Q3), -1.0, 1.0);
        let theta = acos(X);
        let S = sqrt(Q);
        (*res)[0] = -2.0 * S * cos(theta / 3.0) - a / 3.0;
        (*res)[1] = -2.0 * S * cos((theta + 2.0 * PI) / 3.0) - a / 3.0;
        (*res)[2] = -2.0 * S * cos((theta + 4.0 * PI) / 3.0) - a / 3.0;
        return 3;
    } else {
        let alpha = -sgn(R) * pow(abs(R) + sqrt(R2 - Q3), 0.3333);
        var beta = 0.0;
        if (alpha != 0.0) { beta = Q / alpha; }
        (*res)[0] = alpha + beta - a / 3.0;
        return 1;
    }
}

fn qcubic(B: f32, C: f32, D: f32) -> f32 {
    var roots: vec3<f32>;
    let nroots = cubic(1.0, B, C, D, &roots);
    // Sort into descending order.
    if (nroots > 1 && roots.x < roots.y) {
        let swapped = roots.yx;
        roots.x = swapped.x;
        roots.y = swapped.y;
    }
    if (nroots > 2) {
        if (roots.y < roots.z) {
            let swapped = roots.zy;
            roots.y = swapped.x;
            roots.z = swapped.y;
        }
        if (roots.x < roots.y) {
            let swapped = roots.yx;
            roots.x = swapped.x;
            roots.y = swapped.y;
        }
    }
    // Select the largest; nudge a missed near-zero double root, then polish.
    var psi = roots[0];
    if (psi < 0.0) { debug_assert(evalcubic(psi, 1.0, B, C, D) < 0.0); }
    psi = max(1e-6, psi);
    for (var i: i32 = 0; i < 3; i++) {
        let delta = evalcubic(psi, 1.0, B, C, D) / evalquadratic(psi, 3.0, 2.0 * B, C);
        psi -= delta;
    }
    return psi;
}

// The Lanczos quartic method.
fn lquartic(c1: f32, c2: f32, c3: f32, c4: f32, res: ptr<function, vec4<f32>>) -> i32 {
    let alpha = 0.5 * c1;
    let A = c2 - alpha * alpha;
    let B = c3 - alpha * A;
    var a: f32;
    var b: f32;
    var beta: f32;
    var psi: f32;
    psi = qcubic(2.0 * A - alpha * alpha, A * A + 2.0 * B * alpha - 4.0 * c4, -B * B);
    psi = max(0.0, psi);
    a = sqrt(psi);
    beta = 0.5 * (A + psi);
    if (psi <= 0.0) {
        b = sqrt(max(beta * beta - c4, 0.0));
    } else {
        b = 0.5 * a * (alpha - B / psi);
    }
    var res_xy = (*res).xy;
    var resn = quadratic(1.0, alpha + a, beta + b, &res_xy);
    (*res).x = res_xy.x;
    (*res).y = res_xy.y;
    var tmp: vec2<f32>;
    if (quadratic(1.0, alpha - a, beta - b, &tmp) != 0) {
        let previous_xy = (*res).xy;
        (*res).z = previous_xy.x;
        (*res).w = previous_xy.y;
        (*res).x = tmp.x;
        (*res).y = tmp.y;
        resn += 2;
    }
    return resn;
}

fn quartic(A: f32, B: f32, C: f32, D: f32, E: f32, roots: ptr<function, vec4<f32>>) -> i32 {
    var nroots: i32;
    // Solve for the smallest cubic term, retaining the original choice.
    if (abs(B / A) < abs(D / E)) {
        nroots = lquartic(B / A, C / A, D / A, E / A, roots);
    } else {
        nroots = lquartic(D / E, C / E, B / E, A / E, roots);
        for (var i: i32 = 0; i < nroots; i++) {
            (*roots)[i] = 1.0 / (*roots)[i];
        }
    }
    debug_assert(nroots == 0 || nroots == 2 || nroots == 4);
    return nroots;
}

struct Surface {
    params: vec4<f32>,
    p: vec3<f32>,
    colorscheme: i32,
};

// Equation: pp.pp + k(p.p)^2 + k'a^2(p.p) + k''a^4 = 0.
// Expansion for p+tr; the source shifts p so that p.r = 0.
fn goursatsurface(surface: Surface, p: vec3<f32>, r: vec3<f32>, roots: ptr<function, vec4<f32>>) -> i32 {
    let k = surface.params[0];
    let k1 = surface.params[1];
    let k2 = surface.params[2];
    let a = surface.params[3];
    let pp = p * p;
    let pr = p * r;
    let rr = r * r;
    let p2 = dot(p, p);
    let a2 = a * a;
    let a4 = a2 * a2;
    // Inactive source #if 0 general expansion (not activated by this port):
    // let pdr = dot(p, r);
    // let A = dot(rr, rr) + k;
    // let B = 4.0 * dot(pr, rr) + 4.0 * k * pdr;
    // let C = 6.0 * dot(pp, rr) + k * (4.0*pdr*pdr + 2.0*p2) + k1*a2;
    // let D = 4.0 * dot(pp, pr) + 4.0*k*(p2*pdr) + 2.0*k1*a2*pdr;
    // let E = dot(pp, pp) + k*p2*p2 + k1*a2*p2 + k2*a4;
    // Original active expansion, with p.r = 0.
    let A = dot(rr, rr) + k;
    let B = 4.0 * dot(pr, rr);
    let C = 6.0 * dot(pp, rr) + 2.0 * k * p2 + k1 * a2;
    let D = 4.0 * dot(pp, pr);
    let E = dot(pp, pp) + k * p2 * p2 + k1 * a2 * p2 + k2 * a4;
    return quartic(A, B, C, D, E, roots);
}

fn goursatnormal(surface: Surface, p: vec3<f32>) -> vec3<f32> {
    let k = surface.params[0];
    let k1 = surface.params[1];
    let a = surface.params[3];
    return 4.0 * p * p * p + 4.0 * k * dot(p, p) * p + 2.0 * k1 * a * a * p;
}

fn applylighting(baseColor: vec3<f32>, p: vec3<f32>, n_in: vec3<f32>, r: vec3<f32>) -> vec3<f32> {
    var n = n_in;
    if (dot(r, n) > 0.0) { n = -n; }
    var c = baseColor * ambient;
    c += baseColor * diffuse * max(0.0, dot(light, n));
    let s = pow(max(0.0, dot(reflect(light, n), r)), specularpow);
    c += specular * s * specularcolor;
    return c;
}

struct Result {
    p: vec3<f32>,
    n: vec3<f32>,
    basecolor: vec3<f32>,
    t: f32,
};

fn gridline(p: vec3<f32>) -> f32 {
    var t = fract(p * 4.0);
    t = min(t, vec3<f32>(1.0) - t);
    let d = min(t.x, min(t.y, t.z));
    return smoothstep(0.02, 0.025, d);
}

fn dosurface(surface: Surface, p0: vec3<f32>, r: vec3<f32>, roots: ptr<function, vec4<f32>>) -> i32 {
    return goursatsurface(surface, p0, r, roots);
}

fn getnormal(surface: Surface, p: vec3<f32>) -> vec3<f32> {
    return goursatnormal(surface, p);
}

fn solve(surface: Surface, p_in: vec3<f32>, r: vec3<f32>, tmin: f32, result: ptr<function, Result>) -> bool {
    var p = p_in;
    var roots: vec4<f32>;
    let nroots = dosurface(surface, p, r, &roots);
    var t = (*result).t;
    for (var i: i32 = 0; i < 4; i++) {
        if (i == nroots) { break; }
        if (roots[i] > tmin && roots[i] < t) { t = roots[i]; }
    }
    if (t == (*result).t) { return false; }
    p += t * r;
    var n = getnormal(surface, p);
    if (dot(n, r) > 0.0) { n = -n; }
    n = normalize(n);
    var basecolor = abs(n);
    if (surface.colorscheme == 1) { basecolor *= gridline(p); }
    *result = Result(p, n, basecolor, t);
    return true;
}

// https://www.mathcurve.com/surfaces.gb/goursat/goursat.shtml
fn goursatparams(i: i32) -> vec4<f32> {
    var index: i32 = 0;
    var matches = i == index;
    index++;
    if (matches) { return vec4<f32>(0.0, -1.0, 0.0, 1.0); }
    matches = i == index;
    index++;
    if (matches) { return vec4<f32>(0.0, -2.0, 2.0, 1.0); }
    matches = i == index;
    index++;
    if (matches) { return vec4<f32>(-1.0, 1.0, 1.0, 1.0); }
    matches = i == index;
    index++;
    if (matches) { return vec4<f32>(-1.0, -0.25, 0.25, 1.0); }
    matches = i == index;
    index++;
    if (matches) { return vec4<f32>(-0.5, -1.0, 0.5, 1.0); }
    matches = i == index;
    index++;
    if (matches) { return vec4<f32>(-0.5, 1.0, -1.5, 1.0); }
    matches = i == index;
    index++;
    if (matches) { return vec4<f32>(-1.0, 4.0, -6.0, 1.0); }
    matches = i == index;
    index++;
    if (matches) { return vec4<f32>(-1.0, 1.0, 1.0, 1.0); }
    matches = i == index;
    index++;
    if (matches) { return vec4<f32>(-1.0, 2.0, -2.0, 1.0); }
    else { return vec4<f32>(-0.333, -0.666, 0.666, 1.0); }
}

const nparams: i32 = 10;

fn imod(n: i32, m: i32) -> i32 {
    return n - n / m * m;
}

fn scene(p0_in: vec3<f32>, r: vec3<f32>, col: ptr<function, vec3<f32>>) -> bool {
    var p0 = p0_in;
    let tmin = -dot(p0, r);
    p0 += tmin * r;
    var res = Result(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 1e8);
    var ttime = 0.2 * params.time;
    let rtime0 = floor(ttime);
    let rtime1 = floor(ttime + 0.5);
    ttime = fract(2.0 * ttime);
    var surface_params = vec4<f32>(0.0);
    let selection: i32 = i32(MOUSE.w) % (nparams + 1);
    if (selection == 0) {
        let index0: i32 = i32(rtime0) % nparams;
        let index1: i32 = i32(rtime1) % nparams;
        surface_params = mix(goursatparams(index0), goursatparams(index1), ttime);
    } else {
        surface_params = goursatparams(selection - 1);
    }
    let surface = Surface(surface_params, vec3<f32>(0.0), 1);
    if (!solve(surface, p0, r, -tmin, &res)) { return false; }
    *col = applylighting(res.basecolor, res.p, res.n, r);
    return true;
}

fn transform(p_in: vec3<f32>) -> vec3<f32> {
    var p = p_in;
    if (dorotate) {
        let t = params.time;
        let yz = rotate(p.yz, 0.1 * t);
        p.y = yz.x;
        p.z = yz.y;
        let zx = rotate(p.zx, 0.222 * t);
        p.z = zx.x;
        p.x = zx.y;
    }
    return p;
}

fn mainFun(p_in: vec3<f32>, r_in: vec3<f32>, rcentre_in: vec3<f32>) -> vec4<f32> {
    var p = p_in;
    var r = r_in;
    var rcentre = rcentre_in;
    p = transform(p);
    r = normalize(transform(r));
    rcentre = normalize(transform(rcentre));
    let drdx = dpdx(r);
    let drdy = dpdy(r);
    var color = vec4<f32>(0.0);
    let k = dot(r, rcentre);
    var aa = select(2.0, 1.0, XR_IS_AR || k < 0.9);
    // Source AA-band debug variable remains inactive: let aa0 = aa;
    if (MOUSE.z > 0.0) { aa = 1.0; }
    for (var i: f32 = 0.0; i < aa; i += 1.0) {
        for (var j: f32 = 0.0; j < aa; j += 1.0) {
            var col1: vec3<f32>;
            if (scene(p, normalize(r + (i - 0.5 * aa) / aa * drdx + (j - 0.5 * aa) / aa * drdy), &col1)) {
                color += vec4<f32>(col1, 1.0);
            }
        }
    }
    color /= aa * aa;
    if (dot(r, rcentre) > 0.9995) { color.b = 1.0; }
    // Source AA-band debug shading remains inactive: color *= sqrt(aa0/3.0);
    if (alert) { color.x = 1.0; }
    return pow(color, vec4<f32>(0.4545));
}

fn mousetransform(p_in: vec3<f32>) -> vec3<f32> {
    var p = p_in;
    if (MOUSE.x > 0.0) {
        let theta = (2.0 * MOUSE.y - params.resolution.y) / params.resolution.y * PI;
        let phi = (2.0 * MOUSE.x - params.resolution.x) / params.resolution.x * PI;
        let yz = rotate(p.yz, theta);
        p.y = yz.x;
        p.z = yz.y;
        let zx = rotate(p.zx, -phi);
        p.z = zx.x;
        p.x = zx.y;
    }
    return p;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    light = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let scale = 1.0;
    let camera = 4.0;
    let uv = scale * (2.0 * frag_pos.xy - params.resolution) / params.resolution.y;
    var p = vec3<f32>(0.0, 0.0, camera);
    var r = vec3<f32>(uv, -2.0);
    var rcentre = vec3<f32>(0.0, 0.0, -1.0);
    p = mousetransform(p);
    r = mousetransform(r);
    rcentre = mousetransform(rcentre);
    let col = mainFun(p, r, rcentre).xyz;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
