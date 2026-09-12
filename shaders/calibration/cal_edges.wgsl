// Calibration-only edge scene: raymarched primitives over a checkerboard plane.

struct Params {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
};
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

const MARCH_STEPS: i32 = 64;
const CHECKER_FREQUENCY: f32 = 4.0;
const SURFACE_EPSILON: f32 = 0.0015;

fn sd_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sd_box(p: vec3<f32>, half_size: vec3<f32>) -> f32 {
    let q = abs(p) - half_size;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn scene(p: vec3<f32>) -> vec2<f32> {
    let sphere = sd_sphere(p - vec3<f32>(-0.78, 0.82, 0.02), 0.82);
    let box = sd_box(p - vec3<f32>(0.92, 0.58, 0.10), vec3<f32>(0.58, 0.58, 0.58));
    let plane = p.y;
    var nearest = vec2<f32>(sphere, 2.0);
    if (box < nearest.x) {
        nearest = vec2<f32>(box, 3.0);
    }
    if (plane < nearest.x) {
        nearest = vec2<f32>(plane, 1.0);
    }
    return nearest;
}

fn scene_distance(p: vec3<f32>) -> f32 {
    return scene(p).x;
}

fn surface_normal(p: vec3<f32>) -> vec3<f32> {
    let e = 0.001;
    let x = scene_distance(p + vec3<f32>(e, 0.0, 0.0)) - scene_distance(p - vec3<f32>(e, 0.0, 0.0));
    let y = scene_distance(p + vec3<f32>(0.0, e, 0.0)) - scene_distance(p - vec3<f32>(0.0, e, 0.0));
    let z = scene_distance(p + vec3<f32>(0.0, 0.0, e)) - scene_distance(p - vec3<f32>(0.0, 0.0, e));
    return normalize(vec3<f32>(x, y, z));
}

fn hard_shadow(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    var travel = 0.02;
    for (var i = 0; i < 40; i = i + 1) {
        let distance_to_scene = scene_distance(origin + direction * travel);
        if (distance_to_scene < 0.002) {
            return 0.0;
        }
        travel = travel + clamp(distance_to_scene, 0.015, 0.40);
        if (travel > 12.0) {
            break;
        }
    }
    return 1.0;
}

fn material_color(material: f32, p: vec3<f32>) -> vec3<f32> {
    if (material < 1.5) {
        let cell = i32(floor(p.x * CHECKER_FREQUENCY) + floor(p.z * CHECKER_FREQUENCY));
        return select(vec3<f32>(0.08, 0.09, 0.11), vec3<f32>(0.86, 0.88, 0.90), (cell & 1) == 1);
    }
    if (material < 2.5) {
        return vec3<f32>(0.82, 0.19, 0.10);
    }
    return vec3<f32>(0.10, 0.36, 0.86);
}

@fragment
fn fs_main(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let screen = (frag_pos.xy - 0.5 * params.resolution) / params.resolution.y;
    let ray_origin = vec3<f32>(0.0, 2.35, 5.2);
    let camera_target = vec3<f32>(0.0, 0.55, 0.0);
    let forward = normalize(camera_target - ray_origin);
    let right = normalize(cross(forward, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, forward);
    let ray_direction = normalize(forward + screen.x * right + screen.y * up);

    var travel = 0.0;
    var material = 0.0;
    var hit = false;
    for (var i = 0; i < MARCH_STEPS; i = i + 1) {
        let sample_point = ray_origin + ray_direction * travel;
        let sample_value = scene(sample_point);
        if (sample_value.x < SURFACE_EPSILON) {
            material = sample_value.y;
            hit = true;
            break;
        }
        travel = travel + sample_value.x;
        if (travel > 18.0) {
            break;
        }
    }

    var col = mix(vec3<f32>(0.82, 0.90, 0.98), vec3<f32>(0.22, 0.43, 0.74), clamp(0.5 + ray_direction.y, 0.0, 1.0));
    if (hit) {
        let p = ray_origin + ray_direction * travel;
        let normal = surface_normal(p);
        let light_direction = normalize(vec3<f32>(-0.55, 0.82, 0.38));
        var diffuse = max(dot(normal, light_direction), 0.0);
        // FLATLIGHT-SITE
        let shadow = hard_shadow(p + normal * 0.01, light_direction);
        let base = material_color(material, p);
        col = base * (0.16 + 0.84 * diffuse * shadow);
        col = mix(col, vec3<f32>(0.65, 0.76, 0.90), 1.0 - exp(-0.012 * travel * travel));
    }

    // SAFE-BEGIN
    var ballast = 0.0;
    for (var i = 0; i < 5; i = i + 1) {
        let angle = travel * 2.71 + f32(i) * 0.917;
        ballast = ballast + sin(angle) * sin(angle) + cos(angle) * cos(angle) - 1.0;
    }
    col = col + vec3<f32>(ballast * 0.0000001);
    // SAFE-END

    // POSTERIZE-SITE
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
