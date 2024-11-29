const THREAD_COUNT = 16;
const PI = 3.1415927f;
const MAX_DIST = 1000.0;

@group(0) @binding(0)  
  var<storage, read_write> fb : array<vec4f>;

@group(1) @binding(0)
  var<storage, read_write> uniforms : array<f32>;

@group(2) @binding(0)
  var<storage, read_write> shapesb : array<shape>;

@group(2) @binding(1)
  var<storage, read_write> shapesinfob : array<vec4f>;

struct shape {
  transform : vec4f, // xyz = position
  radius : vec4f, // xyz = scale, w = global scale
  rotation : vec4f, // xyz = rotation
  op : vec4f, // x = operation, y = k value, z = repeat mode, w = repeat offset
  color : vec4f, // xyz = color
  animate_transform : vec4f, // xyz = animate position value (sin amplitude), w = animate speed
  animate_rotation : vec4f, // xyz = animate rotation value (sin amplitude), w = animate speed
  quat : vec4f, // xyzw = quaternion
  transform_animated : vec4f, // xyz = position buffer
};

struct march_output {
  color : vec3f,
  depth : f32,
  outline : bool,
};

fn op_smooth_union(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
    var h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    var d = mix(d2, d1, h) - k * h * (1.0 - h);
    var color = mix(col2, col1, h);
    return vec4f(color, d);
}

fn op_smooth_subtraction(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
    // Controlar a suavização com uma interpolação
    var h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    var d = mix(d1, -d2, h) + k * h * (1.0 - h);
    var color = mix(col1, col2, h);

    // Garantir que a subtração mantenha distâncias positivas
    if (d < 0.0) {
        d = 0.0;
    }

    return vec4f(color, d);
}

fn op_smooth_intersection(d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
    var res = op_smooth_union(-d1, -d2, col1, col2, k);
    return vec4f(res.xyz, -res.w);
}

fn op(op: f32, d1: f32, d2: f32, col1: vec3f, col2: vec3f, k: f32) -> vec4f
{
  // union
  if (op < 1.0)
  {
    return op_smooth_union(d1, d2, col1, col2, k);
  }

  // subtraction
  if (op < 2.0)
  {
    return op_smooth_subtraction(d2, d1, col2, col1, k);
  }

  // intersection
  return op_smooth_intersection(d2, d1, col2, col1, k);
}

fn repeat(p: vec3f, offset: vec3f) -> vec3f
{
    return modc(p + 0.5 * offset, offset) - 0.5 * offset;
}

fn transform_p(p: vec3f, option: vec2f) -> vec3f
{
    // Normal mode
    if (option.x <= 1.0) {
        return p;
    }

    // Repeat mode
    var offset = vec3f(option.y);
    return repeat(p, offset);
}

fn scene(p: vec3f) -> vec4f // xyz = color, w = distance
{
    var spheresCount = i32(uniforms[2]);
    var boxesCount = i32(uniforms[3]);
    var torusCount = i32(uniforms[4]);
    var dist = mix(100.0, p.y, uniforms[17]);

    var all_objects_count = spheresCount + boxesCount + torusCount;
    var result = vec4f(vec3f(1.0), dist);

    for (var i = 0; i < all_objects_count; i = i + 1) {
        var shape_info = shapesinfob[i];
        var shape = shapesb[i32(shape_info.y)];

        // 1. Subtrair posição animada
        var p_transformed = p - shape.transform_animated.xyz;

        // 2. Aplicar rotação usando quaternion
        var inv_quat = q_inverse(shape.quat);
        p_transformed = rotate_vector(p_transformed, inv_quat);

        // 3. Aplicar repetição (se necessário)
        p_transformed = transform_p(p_transformed, vec2f(shape.op.z, shape.op.w));

        var dist = 0.0;

        // Determinar tipo de SDF
        if (shape_info.x < 1.0) { // Esfera
            dist = sdf_sphere(p_transformed, shape.radius, vec4f(0.0, 0.0, 0.0, 1.0));
        } else if (shape_info.x < 2.0) { // Caixa arredondada
            dist = sdf_round_box(p_transformed, shape.radius.xyz, shape.radius.w, vec4f(0.0, 0.0, 0.0, 1.0));
        } else if (shape_info.x < 3.0) { // Torus
            dist = sdf_torus(p_transformed, vec2f(shape.radius.x, shape.radius.y), vec4f(0.0, 0.0, 0.0, 1.0));
        }

        // Combine os objetos usando operações booleanas suaves
        result = op(shape.op.x, result.w, dist, result.xyz, shape.color.xyz, shape.op.y);
    }

    return result;
}

fn march(ro: vec3f, rd: vec3f) -> march_output
{
    var max_marching_steps = i32(uniforms[5]);
    var EPSILON = uniforms[23];

    var depth = 0.0;
    var color = vec3f(0.0);
    var hit = false;

    for (var i = 0; i < max_marching_steps; i = i + 1) {
        var pos = ro + depth * rd;
        var res = scene(pos);
        depth += res.w;
        color = res.xyz;
        if (res.w < EPSILON || depth >= MAX_DIST) {
            hit = true;
            break;
        }
    }

    return march_output(color, depth, hit);
}

fn get_normal(p: vec3f) -> vec3f
{
    var e = 0.0005;
    var x = vec3f(e, 0.0, 0.0);
    var y = vec3f(0.0, e, 0.0);
    var z = vec3f(0.0, 0.0, e);

    var nx = scene(p + x).w - scene(p - x).w;
    var ny = scene(p + y).w - scene(p - y).w;
    var nz = scene(p + z).w - scene(p - z).w;

    var n = normalize(vec3f(nx, ny, nz));

    return n;
}

// https://iquilezles.org/articles/rmshadows/
fn get_soft_shadow(ro: vec3f, rd: vec3f, tmin: f32, tmax: f32, k: f32) -> f32
{
    var res = 1.0;
    var t = tmin;
    var EPSILON = 0.0001;

    for (var i = 0; i < 128 && t < tmax; i = i + 1)
    {
        var h = scene(ro + rd * t).w;
        if (h < EPSILON)
        {
            return 0.0; // Em sombra
        }
        res = min(res, k * h / t);
        t += clamp(h, 0.005, 1.0);
    }
    return clamp(res, 0.0, 1.0);
}

fn get_AO(current: vec3f, normal: vec3f) -> f32
{
  var occ = 0.0;
  var sca = 1.0;
  for (var i = 0; i < 5; i = i + 1)
  {
    var h = 0.001 + 0.15 * f32(i) / 4.0;
    var d = scene(current + h * normal).w;
    occ += (h - d) * sca;
    sca *= 0.95;
  }

  return clamp( 1.0 - 2.0 * occ, 0.0, 1.0 ) * (0.5 + 0.5 * normal.y);
}

fn get_ambient_light(light_pos: vec3f, sun_color: vec3f, rd: vec3f) -> vec3f
{
  var backgroundcolor1 = int_to_rgb(i32(uniforms[12]));
  var backgroundcolor2 = int_to_rgb(i32(uniforms[29]));
  var backgroundcolor3 = int_to_rgb(i32(uniforms[30]));
  
  var ambient = backgroundcolor1 - rd.y * rd.y * 0.5;
  ambient = mix(ambient, 0.85 * backgroundcolor2, pow(1.0 - max(rd.y, 0.0), 4.0));

  var sundot = clamp(dot(rd, normalize(vec3f(light_pos))), 0.0, 1.0);
  var sun = 0.25 * sun_color * pow(sundot, 5.0) + 0.25 * vec3f(1.0,0.8,0.6) * pow(sundot, 64.0) + 0.2 * vec3f(1.0,0.8,0.6) * pow(sundot, 512.0);
  ambient += sun;
  ambient = mix(ambient, 0.68 * backgroundcolor3, pow(1.0 - max(rd.y, 0.0), 16.0));

  return ambient;
}

fn get_light(current: vec3f, obj_color: vec3f, rd: vec3f) -> vec3f
{
    var light_position = vec3f(uniforms[13], uniforms[14], uniforms[15]);
    var sun_color = int_to_rgb(i32(uniforms[16]));
    var ambient = get_ambient_light(light_position, sun_color, rd);
    var normal = get_normal(current);

    if (length(current) > uniforms[20] + uniforms[8]) {
        return ambient;
    }

    var light_dir = normalize(light_position - current);
    var diff = clamp(dot(normal, light_dir), 0.0, 1.0);
    var shadow = get_soft_shadow(current, light_dir, 0.01, 20.0, 5.0);
    var irradiance = shadow * diff;
    var ao = get_AO(current, normal);

    return ambient * ao * obj_color + obj_color * sun_color * irradiance;
}

fn set_camera(ro: vec3f, ta: vec3f, cr: f32) -> mat3x3<f32>
{
  var cw = normalize(ta - ro);
  var cp = vec3f(sin(cr), cos(cr), 0.0);
  var cu = normalize(cross(cw, cp));
  var cv = normalize(cross(cu, cw));
  return mat3x3<f32>(cu, cv, cw);
}

fn animate(val: vec3f, time_scale: f32, offset: f32) -> vec3f
{
    var time = uniforms[0];
    return val * sin(time * time_scale + offset);}

@compute @workgroup_size(THREAD_COUNT, 1, 1)
fn preprocess(@builtin(global_invocation_id) id : vec3u)
{
    var time = uniforms[0];
    var spheresCount = i32(uniforms[2]);
    var boxesCount = i32(uniforms[3]);
    var torusCount = i32(uniforms[4]);
    var all_objects_count = spheresCount + boxesCount + torusCount;

    if (id.x >= u32(all_objects_count)) {
        return;
    }

    var index = i32(id.x);
    var shape = shapesb[index];

    // 1. Atualizar posição animada
    if (shape.animate_transform.w != 0.0) {
        var animated_position = shape.transform.xyz +
                                animate(shape.animate_transform.xyz, shape.animate_transform.w, 0.0);
        shape.transform_animated = vec4f(animated_position, shape.transform.w);
    } else {
        shape.transform_animated = shape.transform;
    }

    // 2. Atualizar quaternion (rotação animada)
    if (shape.animate_rotation.w != 0.0) {
        var animated_rotation = shape.rotation.xyz +
                                animate(shape.animate_rotation.xyz, shape.animate_rotation.w, 0.0);
        shape.quat = quaternion_from_euler(animated_rotation);
    } else {
        shape.quat = quaternion_from_euler(shape.rotation.xyz);
    }

    // Normalize o quaternion
    shape.quat = normalize(shape.quat);

    // Escreva as alterações de volta no buffer
    shapesb[index] = shape;
}

@compute @workgroup_size(THREAD_COUNT, THREAD_COUNT, 1)
fn render(@builtin(global_invocation_id) id : vec3u)
{
    var fragCoord = vec2f(f32(id.x), f32(id.y));
    var rez = vec2f(uniforms[1]);
    var time = uniforms[0];

    var lookfrom = vec3f(uniforms[6], uniforms[7], uniforms[8]);
    var lookat = vec3f(uniforms[9], uniforms[10], uniforms[11]);
    var camera = set_camera(lookfrom, lookat, 0.0);
    var ro = lookfrom;

    var uv = (fragCoord - 0.5 * rez) / rez.y;
    uv.y = -uv.y;
    var rd = camera * normalize(vec3f(uv, 1.0));

    var march_res = march(ro, rd);
    var color: vec3f;

  /*  if (march_res.outline) {
        color = vec3f(0.0);
    } else if (march_res.depth < uniforms[20]) {
        var pos = ro + rd * march_res.depth;
        color = get_light(pos, march_res.color, rd);
        color = vec3f (1.0);
    } else {
        color = get_ambient_light(vec3f(uniforms[13], uniforms[14], uniforms[15]), int_to_rgb(i32(uniforms[16])), rd);
        color = vec3f (1.0, 0,0);

    }

    color = linear_to_gamma(color);

  */
  var pos = ro + rd * march_res.depth;

  color = get_light(pos, march_res.color, rd);

  fb[mapfb(id.xy, uniforms[1])] = vec4f(linear_to_gamma(color), 1.0);
}