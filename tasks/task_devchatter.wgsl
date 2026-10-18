// Original: For DevChatter, by mrange.
// URL: ./sources/round2/mrange_devchatter.glsl (local snapshot;
// upstream permalink was not recorded with the supplied source).
// Source: tasks/sources/round2/mrange_devchatter.glsl.
// License: CC0; exact statement: source lines 1-2,
// "License: CC0 - https://creativecommons.org/share-your-work/public-domain/cc0/".
// Port date: 2026-10-17. Faithful GLSL-to-WGSL port; no optimization.
// Port decision: retain #define AA 1 and activate #elif AA == 1, the 2x2
// supersampling branch. AA == 0 stays inactive; both loops and /4 kept.
// Port changes: house uniform/entry points; floor-mod preserves signed time;
// atan(y,x) becomes atan2; ternaries become select; GLSL vector equality uses
// all(). Out/inout material and march outputs use function pointers; mutable
// inputs use copies. Rotation inout helpers return their vector, with original
// swizzle stores expanded to components. logor overloads become logor_rad and
// logor_point; saturate overloads become saturate_v3/v2/f; reserved ref and
// final become reflectivity and result_color. Formerly uninitialized locals
// use WGSL zero initialization; seeded caller outputs are passed by pointer.
// All bounce/march/shadow/normal evaluations, unused helpers, and dead code
// remain in their original roles; optional clamping/time alternatives stay dead.
// Final RGB is clamped to [0,1] to match framebuffer clamp; alpha is 1.0.
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

const TOLERANCE: f32 = 0.0001;
const MAX_RAY_LENGTH: f32 = 16.0;
const MAX_BOUNCES: i32 = 4;
const MAX_RAY_MARCHES: i32 = 48;
const PI: f32 = 3.141592654;
const TAU: f32 = 2.0 * PI;
const DEG2RAD: f32 = PI / 180.0;
const PERIOD: f32 = 12.0;
const FADETIME: f32 = 2.0;
const AA: i32 = 1; // Active source branch is AA == 1: 2x2 samples.

fn mod_f(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}

fn pR(p: vec2<f32>, a: f32) -> vec2<f32> {
    return cos(a) * p + sin(a) * vec2<f32>(p.y, -p.x);
}

fn getTimeInPeriod() -> f32 {
    // Inactive source alternative: return mod_f(params.time/10.0, PERIOD)+3.0;
    return mod_f(params.time, PERIOD);
}

fn getPeriod() -> f32 {
    return floor(params.time / PERIOD);
}

fn maxComp(p: vec3<f32>) -> f32 {
    return max(p.x, max(p.y, p.z));
}

fn lengthN(v_in: vec3<f32>, n: f32) -> f32 {
    var v = v_in;
    v = abs(v);
    v = pow(v, vec3<f32>(n));
    return pow(v.x + v.y + v.z, 1.0 / n);
}

fn sdRoundCube(p: vec3<f32>, r: f32) -> f32 {
    return lengthN(p, 8.0) - r;
}

fn sgn(x: f32) -> f32 {
    return select(1.0, -1.0, x < 0.0);
}

fn pR90(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(p.y, -p.x);
}

fn pR270(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-p.y, p.x);
}

fn sdCapsule(p_in: vec3<f32>, h: vec2<f32>) -> f32 {
    var p = p_in;
    let xy = pR90(p.xy);
    p.x = xy.x;
    p.y = xy.y;
    let r = h.x;
    let c = h.y / 2.0;
    return mix(length(p.xz) - r, length(vec3<f32>(p.x, abs(p.y) - c, p.z)) - r, step(c, abs(p.y)));
}

fn capsule(p_in: vec3<f32>, f: vec2<f32>, t: vec2<f32>, r: f32) -> f32 {
    var p = p_in;
    let diff = t - f;
    let a = atan2(diff.y, diff.x);
    let shifted_xy = p.xy - (f + diff * 0.5);
    p.x = shifted_xy.x;
    p.y = shifted_xy.y;
    let xy = pR(p.xy, a);
    p.x = xy.x;
    p.y = xy.y;
    return sdCapsule(p, vec2<f32>(r, length(diff)));
}

const logos: f32 = 1.0 / 350.0;

fn logor_rad(p: vec3<f32>, rad: f32) -> f32 {
    var r = capsule(p, vec2<f32>(0.0, 0.0), vec2<f32>(228.0, 0.0), rad);
    r = min(r, capsule(p, vec2<f32>(228.0, 0.0), vec2<f32>(228.0 + 140.0, -100.0), rad));
    r = min(r, capsule(p, vec2<f32>(228.0 + 140.0, -100.0), vec2<f32>(228.0 + 140.0 + 150.0, -100.0), rad));
    return r;
}

fn logob(p: vec3<f32>, rad: f32) -> f32 {
    var b = capsule(p, vec2<f32>(36.0, -36.0 - 9.0), vec2<f32>(36.0, -100.0), rad);
    b = min(b, capsule(p, vec2<f32>(36.0, -100.0), vec2<f32>(228.0, -100.0), rad));
    b = min(b, capsule(p, vec2<f32>(228.0, -100.0), vec2<f32>(228.0 + 140.0, 0.0), rad));
    b = min(b, capsule(p, vec2<f32>(228.0 + 140.0, 0.0), vec2<f32>(228.0 + 140.0 + 150.0, 0.0), rad));
    return b;
}

fn ilogo(p_in: vec3<f32>) -> f32 {
    var p = p_in;
    p /= logos;
    p -= vec3<f32>(-250.0, 350.0, 40.0);
    let zy = pR270(p.zy);
    p.z = zy.x;
    p.y = zy.y;
    let r = logor_rad(p, 28.0);
    let b = logob(p, 28.0);
    let d = min(r, b);
    return d * logos;
}

fn linstep(f: f32, t: f32, x: f32) -> f32 {
    return clamp((x - f) / (t - f), 0.0, 1.0);
}

fn logorcolor() -> vec3<f32> {
    let time = getTimeInPeriod();
    let lt = time - 3.0;
    return mix(mix(15.0 * vec3<f32>(1.0, 3.0, 5.0), -3.0 * vec3<f32>(5.0), linstep(0.0, 0.25, lt)),
               10.0 * vec3<f32>(-1.0 / 3.0, 3.0, 5.0), linstep(1.5, 8.0, lt));
}

fn logorradiance() -> vec3<f32> {
    let time = getTimeInPeriod();
    let lt = time - 3.0;
    return mix(mix(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.4, 0.2, 0.1), linstep(0.125, 0.5, lt)),
               vec3<f32>(0.1, 0.0, 0.0), linstep(1.5, 4.0, lt));
}

fn logo(p_in: vec3<f32>, col: ptr<function, vec3<f32>>, reflectivity: ptr<function, f32>,
        trans: ptr<function, f32>, absorb: ptr<function, vec3<f32>>) -> f32 {
    var p = p_in;
    p /= logos;
    p -= vec3<f32>(-250.0, 350.0, 40.0);
    let zy = pR270(p.zy);
    p.z = zy.x;
    p.y = zy.y;
    let r = logor_rad(p, 18.0);
    var b = logob(p, 18.0);
    b = max(b, -capsule(p, vec2<f32>(228.0, 0.0), vec2<f32>(228.0 + 140.0, -100.0), 36.0));
    let d = min(r, b);
    if (d == r) {
        *reflectivity = 0.4;
        *trans = 0.8;
        *col = vec3<f32>(1.0);
        *absorb = logorcolor();
    } else {
        *reflectivity = 0.4;
        *trans = 0.8;
        *col = vec3<f32>(1.0);
        *absorb = 7.0 * vec3<f32>(1.0);
    }
    return d * logos;
}

fn logor_point(p_in: vec3<f32>) -> f32 {
    var p = p_in;
    p /= logos;
    p -= vec3<f32>(-250.0, 350.0, 40.0);
    let zy = pR270(p.zy);
    p.z = zy.x;
    p.y = zy.y;
    return logor_rad(p, 18.0) * logos;
}

fn distanceField(p: vec3<f32>, col: ptr<function, vec3<f32>>, reflectivity: ptr<function, f32>,
                 trans: ptr<function, f32>, absorb: ptr<function, vec3<f32>>) -> f32 {
    let rc = sdRoundCube(p, 1.0);
    let l = logo(p, col, reflectivity, trans, absorb);
    let il = ilogo(p);
    var mat: i32 = 0;
    var d = rc;
    d = max(rc, -il);
    if (d == -il) { mat = 1; }
    d = min(d, l);
    if (d == l) { mat = 2; }
    if (mat == 0) {
        *reflectivity = 0.4;
        *trans = 0.0;
        *col = vec3<f32>(1.0);
    } else if (mat == 1) {
        *reflectivity = 0.05;
        *trans = 0.0;
        *col = vec3<f32>(1.0);
    } else {
        // Keep whatever we got.
    }
    return d;
}

fn saturate_v3(a: vec3<f32>) -> vec3<f32> { return clamp(a, vec3<f32>(0.0), vec3<f32>(1.0)); }
fn saturate_v2(a: vec2<f32>) -> vec2<f32> { return clamp(a, vec2<f32>(0.0), vec2<f32>(1.0)); }
fn saturate_f(a: f32) -> f32 { return clamp(a, 0.0, 1.0); }

const lightPos1: vec3<f32> = 100.0 * vec3<f32>(-0.3, 0.15, 1.0);
const lightPos2: vec3<f32> = 100.0 * vec3<f32>(-1.0, -0.20, -3.0);
const lightCol1: vec3<f32> = vec3<f32>(8.0 / 8.0, 7.0 / 8.0, 6.0 / 8.0);
const lightCol2: vec3<f32> = vec3<f32>(8.0 / 8.0, 6.0 / 8.0, 7.0 / 8.0);

fn getSkyColor(rayDir_in: vec3<f32>) -> vec3<f32> {
    var rayDir = rayDir_in;
    let zy = pR(rayDir.zy, 1.2);
    rayDir.z = zy.x;
    rayDir.y = zy.y;
    let lightDir1 = normalize(lightPos1);
    let lightDir2 = normalize(lightPos2);
    let ld1 = max(dot(lightDir1, rayDir), 0.0);
    let ld2 = max(dot(lightDir2, rayDir), 0.0);
    var result_color = vec3<f32>(0.125);
    if (rayDir.y > abs(rayDir.x) * 1.0 && rayDir.y > abs(rayDir.z * 0.25)) {
        result_color = vec3<f32>(2.0) * rayDir.y;
    }
    let roundBox = length(max(abs(rayDir.xz / max(0.0, rayDir.y)) - vec2<f32>(0.9, 4.0), vec2<f32>(0.0))) - 0.1;
    result_color += vec3<f32>(0.8) * pow(saturate_f(1.0 - roundBox * 0.5), 6.0);
    result_color += pow(lightCol1, vec3<f32>(2.0, 1.5, 1.5)) * pow(ld1, 8.0);
    result_color += lightCol1 * pow(ld1, 200.0);
    result_color += pow(lightCol2, vec3<f32>(2.0, 1.5, 1.5)) * pow(ld2, 1.0);
    result_color += lightCol2 * pow(ld2, 200.0);
    return result_color;
}

fn normal(pos: vec3<f32>) -> vec3<f32> {
    let eps = vec3<f32>(0.001, 0.0, 0.0);
    var col: vec3<f32>;
    var reflectivity: f32;
    var trans: f32;
    var nor: vec3<f32>;
    var absorb: vec3<f32>;
    nor.x = distanceField(pos + eps.xyy, &col, &reflectivity, &trans, &absorb) - distanceField(pos - eps.xyy, &col, &reflectivity, &trans, &absorb);
    nor.y = distanceField(pos + eps.yxy, &col, &reflectivity, &trans, &absorb) - distanceField(pos - eps.yxy, &col, &reflectivity, &trans, &absorb);
    nor.z = distanceField(pos + eps.yyx, &col, &reflectivity, &trans, &absorb) - distanceField(pos - eps.yyx, &col, &reflectivity, &trans, &absorb);
    return normalize(nor);
}

fn rayMarch(dmod: f32, ro: vec3<f32>, rd: ptr<function, vec3<f32>>, mint: f32, maxt: f32,
            rep: ptr<function, i32>, col: ptr<function, vec3<f32>>, reflectivity: ptr<function, f32>,
            trans: ptr<function, f32>, absorb: ptr<function, vec3<f32>>) -> f32 {
    var t = mint;
    for (var i: i32 = 0; i < MAX_RAY_MARCHES; i++) {
        let distance_ = distanceField(ro + *rd * t, col, reflectivity, trans, absorb);
        let dist = dmod * distance_;
        if (dist < TOLERANCE || t > maxt) { break; }
        t += max(dist, 0.001);
        *rep = i;
    }
    return t;
}

fn softShadow(pos: vec3<f32>, ld: vec3<f32>, ll: f32, mint: f32, k: f32) -> f32 {
    let minShadow = 0.25;
    var res = 1.0;
    var t = mint;
    var col: vec3<f32>;
    var reflectivity: f32;
    var trans: f32;
    var absorb: vec3<f32>;
    for (var i: i32 = 0; i < 24; i++) {
        let dist = distanceField(pos + ld * t, &col, &reflectivity, &trans, &absorb);
        res = min(res, k * dist / t);
        if (ll <= t) { break; }
        if (res <= minShadow) { break; }
        t += max(mint * 0.2, dist);
    }
    return clamp(res, minShadow, 1.0);
}

fn postProcess(col_in: vec3<f32>, q: vec2<f32>) -> vec3<f32> {
    var col = col_in;
    // Inactive source: col = saturate_v3(col);
    col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.75));
    col = col * 0.6 + 0.4 * col * col * (vec3<f32>(3.0) - 2.0 * col);
    col = mix(col, vec3<f32>(dot(col, vec3<f32>(0.33))), -0.4);
    col *= 0.5 + 0.5 * pow(19.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.7);
    return col;
}

fn render(ro_in: vec3<f32>, rd_in: vec3<f32>) -> vec3<f32> {
    var ro = ro_in;
    var rd = rd_in;
    let lightPos = 1.4 * vec3<f32>(1.0, 2.5, 2.0);
    var result_color = vec3<f32>(0.0);
    var ragg = vec3<f32>(1.0);
    var tdist = 0.0;
    let refraction = 0.9;
    var inside = false;
    for (var i: i32 = 0; i < MAX_BOUNCES; i++) {
        if (maxComp(ragg) < 0.01) { break; }
        let dmod = select(1.0, -1.0, inside);
        var absorb = vec3<f32>(3.0, 2.0, 1.0);
        var col = vec3<f32>(1.0);
        var reflectivity = 0.2;
        var trans = 0.9;
        var rep: i32 = 0;
        let t = rayMarch(dmod, ro, &rd, 0.01, MAX_RAY_LENGTH, &rep, &col, &reflectivity, &trans, &absorb);
        tdist += t;
        let pos = ro + t * rd;
        var nor = vec3<f32>(0.0, 1.0, 0.0);
        if (t < MAX_RAY_LENGTH) {
            nor = normal(pos);
        } else {
            result_color += ragg * getSkyColor(rd);
            break;
        }
        var mref = refraction;
        if (inside) {
            nor = -nor;
            mref = 1.0 / refraction;
        }
        let refl = reflect(rd, nor);
        let refr = refract(rd, nor, mref);
        let lv = lightPos - pos;
        let ld = normalize(lv);
        let ll = length(lv);
        // Original TODO: rework shadow to work with transparent objects.
        var sha = 1.0;
        if (!inside) { sha = softShadow(pos, ld, ll, 0.01, 64.0); }
        let dif = max(dot(nor, ld), 0.0);
        let occ = 1.0 - f32(rep) / f32(MAX_RAY_MARCHES);
        let l = dif * occ * sha;
        var lr = vec3<f32>(0.0);
        let dr = logor_point(pos);
        if (!inside && dr > TOLERANCE) { lr = logorradiance() / (8.0 * dr); }
        let lin = mix(0.2, 1.0, l);
        let sky = getSkyColor(refl);
        col *= vec3<f32>(0.7) + 0.3 * abs(nor.zxy);
        let mcol = mix(lin * col + lr, sky, reflectivity);
        var beer = vec3<f32>(1.0);
        if (inside) { beer = exp(-absorb * t); }
        result_color += (1.0 - trans) * ragg * beer * mcol;
        ragg *= trans * beer;
        ro = pos;
        if (all(refr == vec3<f32>(0.0))) {
            rd = refl;
        } else {
            rd = refr;
        }
        if (dot(refl, rd) < 0.9) { inside = !inside; }
    }
    return result_color;
}

fn getSample(p: vec2<f32>) -> vec3<f32> {
    let time = getTimeInPeriod();
    let rob = vec3<f32>(-0.05, 1.0, -0.1);
    var ro = (0.65 + time / 16.0) * 2.0 * rob;
    let la = rob;
    let xz = pR(ro.xz, -time / 32.0);
    ro.x = xz.x;
    ro.z = xz.y;
    let ww = normalize(la - ro);
    let uu = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), ww));
    let vv = normalize(cross(ww, uu));
    let rd = normalize(p.x * uu + p.y * vv + 2.0 * ww);
    let col = render(ro, rd);
    return col;
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let time = getTimeInPeriod();
    let q = frag_pos.xy / params.resolution;
    var p = vec2<f32>(-1.0) + 2.0 * q;
    p.x *= params.resolution.x / params.resolution.y;
    // Inactive source AA == 0: let col = getSample(p);
    // Active source #elif AA == 1, with its original sample placement.
    var col = vec3<f32>(0.0);
    let unit = vec2<f32>(1.0) / params.resolution;
    for (var y: i32 = 0; y < 2; y++) {
        for (var x: i32 = 0; x < 2; x++) {
            col += getSample(p - 0.5 * unit + unit * vec2<f32>(f32(x), f32(y)));
        }
    }
    col /= 4.0;
    let fadeIn = smoothstep(0.0, FADETIME, time);
    let fadeOut = smoothstep(PERIOD - FADETIME, PERIOD, time);
    col = mix(postProcess(col, q) * fadeIn, vec3<f32>(1.0), fadeOut);
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
