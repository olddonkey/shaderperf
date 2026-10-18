// common stuff
const vec3 fv3_1    = vec3(1.0, 1.0, 1.0);
const vec3 fv3_0    = vec3(0.0, 0.0, 0.0);
const vec3 fv3_x    = vec3(1.0, 0.0, 0.0);
const vec3 fv3_y    = vec3(0.0, 1.0, 0.0);
const vec3 fv3_z    = vec3(0.0, 0.0, 1.0);
const vec2 fv2_1    = vec2(1.0, 1.0);
const vec2 fv2_0    = vec2(0.0, 0.0);
const vec2 fv2_x    = vec2(1.0, 0.0);
const vec2 fv2_y    = vec2(0.0, 1.0);
const float PI        = 3.14159265359;
const float TAU     = PI * 2.0;

vec2 complexMul(in vec2 A, in vec2 B) {
    return vec2((A.x * B.x) - (A.y * B.y), (A.x * B.y) + (A.y * B.x));
}

struct POI {
    vec2    center;
    float range;
    float maxIter;
};

float sdfTrapOrigin(in vec2 p) {
    return max(0.0, length(p) - 0.0);
}

float sdfTrapX(in vec2 p) {
    return p.x;
}

float sdfTrapY(in vec2 p) {
    return p.y;
}

#define sdfTRAP sdfTrapX

void mandelTrapCumDist(in vec2 C, float maxIters, out vec2 trapDist) {
    vec2  Z  = C;
    trapDist = vec2(0.0);
    float c = cos(iTime * 0.1);
    float s = sin(iTime * 0.1);
    mat2 rot = mat2(s, c, -c, s);
    for (float n = 0; n < maxIters; n += 1.0) {
        Z    = complexMul(Z, Z) + C;
        vec2 z = Z * rot;
        trapDist += 1.0 / vec2(sdfTrapX(z), sdfTrapY(z));
    }
}

void mandelTrapMinDist(in vec2 C, float maxIters, out vec2 trapDist) {
    vec2  Z  = C;
    trapDist = vec2(1e-20);
    float c = cos(iTime * 0.1);
    float s = sin(iTime * 0.1);
    mat2 rot = mat2(s, c, -c, s);
    for (float n = 0; n < maxIters; n += 1.0) {
        Z    = complexMul(Z, Z) + C;
        vec2 z = Z * rot;
        vec2 d = vec2(sdfTrapX(z), sdfTrapY(z));
        if (abs(d.x) > abs(trapDist.x)) {
            trapDist.x = d.x;
        }
        if (abs(d.y) > abs(trapDist.y)) {
            trapDist.y = d.y;
        }
  //      trapDist = min(trapDist, vec2(sdfTrapX(z), sdfTrapY(z)));
    }
}

float mandelEscapeIters(in vec2 C, in float maxIters) {
    vec2 Z = C;
    for (float n = 0; n < maxIters; n += 1.0) {
        Z    = complexMul(Z, Z) + C;
        if (dot(Z, Z) > 4.0) {
            return n;
        }
    }
    return maxIters;
}

void mainImage(out vec4 RGBA, in vec2 XY)
{
    RGBA.a  = 1.0;
    float smallWay = min(iResolution.x, iResolution.y);
    vec2 uv = (XY * 2.0 - fv2_1 * iResolution.xy)/smallWay;
    float t = iTime * TAU / 5.0;

    const POI poi1 = POI(vec2(-.7105, 0.2466), 0.004, 400.0);
    const POI poi2 = POI(vec2(-.7500, 0.0000), 1.000, 50.0);
    const POI poi  = poi2;

    float rng = poi.range * (1.0 + sin(t) * 0.1);
    
    vec3 rgb;

    vec2  C   = uv * rng + poi.center;
//  float f   = 1.0 - mandelEscapeIters(C, poi.maxIter) / poi.maxIter;
    vec2  td;
    // mandelTrapCumDist(C, poi.maxIter, td);
    mandelTrapMinDist(C, poi.maxIter, td);
    if (length(td) < 2.0) {
        vec2 f = td;
        f *= 4.0;
        f = sin(f) * 0.5 + 0.5;
        rgb = vec3(f, 0.5);
    }
    else {
        rgb = vec3(sin(length(C) * 80.0) * 0.03 + 0.5);
    }
//    f = fract(f);
  //  f = 1.0 / f;
  //  f *= poi.maxIter * 3.0;
    //f = fract(f);
 //   f = sin(pow(f, vec2(0.5)) * poi.maxIter) * 0.5 + 0.5;
 //   f = pow(f, 2.0) * 100000.0;

    RGBA.rgb = rgb;
}

// grimoire bindings
out vec4 fragColor; void main() { mainImage(fragColor, gl_FragCoord.xy); }
