
fn linear_to_gamma_channel(channel: f32) -> f32
{
  return pow(channel, 0.4545);
}

fn random(p: vec2f) -> f32
{
  return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

fn linear_to_gamma(color: vec3f) -> vec3f
{
  return vec3f(linear_to_gamma_channel(color.x), linear_to_gamma_channel(color.y), linear_to_gamma_channel(color.z));
}

fn rot(a: f32) -> mat2x2<f32>
{
  var s = sin(a);
  var c = cos(a);
  return mat2x2<f32>(c, -s, s, c);
}

fn degrees_to_radians(degrees: f32) -> f32
{
  return degrees * PI / 180.0;
}

fn mapfb(p: vec2u, rez: f32) -> i32
{
  return i32(p.x) + i32(p.y) * i32(rez);
}

fn int_to_rgb(c: i32) -> vec3f
{
  var r = f32((c >> 16) & 0xff) / 255.0;
  var g = f32((c >> 8) & 0xff) / 255.0;
  var b = f32(c & 0xff) / 255.0;

  return vec3f(r, g, b);
}

fn rgb_to_int(c: vec4f) -> i32
{
  var r = i32(c.x * 255.0) << 16;
  var g = i32(c.y * 255.0) << 8;
  var b = i32(c.z * 255.0);

  return r | g | b;
}

fn quaternion_from_euler(euler: vec3f) -> vec4f
{
  var c1 = cos(euler.x / 2.0);
  var c2 = cos(euler.y / 2.0);
  var c3 = cos(euler.z / 2.0);
  var s1 = sin(euler.x / 2.0);
  var s2 = sin(euler.y / 2.0);
  var s3 = sin(euler.z / 2.0);

  var q = vec4f(0.0);
  q.x = s1 * c2 * c3 + c1 * s2 * s3;
  q.y = c1 * s2 * c3 - s1 * c2 * s3;
  q.z = c1 * c2 * s3 + s1 * s2 * c3;
  q.w = c1 * c2 * c3 - s1 * s2 * s3;

  return q;
}

fn modc(a: vec3f, b: vec3f) -> vec3f
{
  return a - b * floor(a / b);
}

fn qmul(q1: vec4f, q2: vec4f) -> vec4f
{
  return vec4f(
    q2.xyz * q1.w + q1.xyz * q2.w + cross(q1.xyz, q2.xyz),
    q1.w * q2.w - dot(q1.xyz, q2.xyz)
  );
}

fn q_conjugate(q: vec4f) -> vec4f
{
  return vec4f(-q.xyz, q.w);
}

fn q_inverse(q: vec4f) -> vec4f
{
  return q_conjugate(q) / dot(q, q);
}

fn rotate_vector(v: vec3f, r: vec4f) -> vec3f
{
  var r_c = r * vec4f(-1.0, -1.0, -1.0, 1.0);
  return qmul(r, qmul(vec4f(v, 0.0), r_c)).xyz;
}