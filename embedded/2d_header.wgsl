// 2D Header with utility functions. Cannot be replaced from the API

struct U2D {
  background: vec4<f32>,
  fb_size: vec2<u32>, _pad0: vec2<u32>,
  gp0: vec4<f32>, gp1: vec4<f32>, gp2: vec4<f32>,
  gp3: vec4<f32>, gp4: vec4<f32>, gp5: vec4<f32>,
  gp6: vec4<f32>, gp7: vec4<f32>, gp8: vec4<f32>, gp9: vec4<f32>,
  mat2d_c0: vec4<f32>,
  mat2d_c1: vec4<f32>,
  mat2d_c2: vec4<f32>,
  mat2d_inv_c0: vec4<f32>,
  mat2d_inv_c1: vec4<f32>,
  mat2d_inv_c2: vec4<f32>,
  lights_count: u32,
  _pad_lights: vec3<u32>,
};

@group(0) @binding(0) var<uniform> U: U2D;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var atlas_tex: texture_2d<f32>;
@group(0) @binding(3) var atlas_smp: sampler;
struct Vert { pos: vec2<f32>, uv: vec2<f32> };
struct Verts { data: array<Vert> };
struct Indices { data: array<u32> };
@group(0) @binding(4) var<storage, read> verts: Verts;
@group(0) @binding(5) var<storage, read> indices: Indices;
struct U32s { data: array<u32> };
@group(0) @binding(6) var<storage, read> tile_offsets: U32s;
@group(0) @binding(7) var<storage, read> tile_counts:  U32s;
@group(0) @binding(8) var<storage, read> tile_tris:    U32s;

struct LightWGSL {
  header:   vec4<u32>,  // [light_type, emitting, _, _]
  position: vec4<f32>,  // xyz, _
  color:    vec4<f32>,  // rgb, _
  params0:  vec4<f32>,  // [intensity, radius, startD, endD]
  params1:  vec4<f32>,  // [flicker, _, _, _]
};
struct Lights { data: array<LightWGSL>, };
@group(0) @binding(9) var<storage, read> lights: Lights;
@group(0) @binding(10) var atlas_mat_tex: texture_2d<f32>;

fn tiles_x() -> u32 { return (U.fb_size.x + 7u) / 8u; }
fn tiles_y() -> u32 { return (U.fb_size.y + 7u) / 8u; }
fn tile_index(tx: u32, ty: u32) -> u32 { return ty * tiles_x() + tx; }
fn tile_of_px(px: u32, py: u32) -> u32 {
  let tx = px / 8u;
  let ty = py / 8u;
  return tile_index(tx, ty);
}

fn sv_write(px: u32, py: u32, c: vec4<f32>) {
  textureStore(color_out, vec2<i32>(i32(px), i32(py)), c);
}
fn sv_sample(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(atlas_tex, atlas_smp, uv, 0.0);
}
// ----- SceneVM 2D helpers -----
struct BaryHit { hit: bool, w: vec3<f32> };
struct ColorHit { hit: bool, color: vec4<f32>, tri: u32, uv: vec2<f32> };

fn sv_edge(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  return (p.x - a.x)*(b.y - a.y) - (p.y - a.y)*(b.x - a.x);
}

fn sv_tri_bary(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> BaryHit {
  let e0 = sv_edge(p,a,b);
  let e1 = sv_edge(p,b,c);
  let e2 = sv_edge(p,c,a);
  let ok = (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0);
  if (!ok) { return BaryHit(false, vec3<f32>(0.0)); }
  let area = abs((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x));
  if (area <= 0.0) { return BaryHit(false, vec3<f32>(0.0)); }
  let w0 = abs((b.x - p.x)*(c.y - p.y) - (b.y - p.y)*(c.x - p.x)) / area;
  let w1 = abs((c.x - p.x)*(a.y - p.y) - (c.y - p.y)*(a.x - p.x)) / area;
  let w2 = 1.0 - w0 - w1;
  return BaryHit(true, vec3<f32>(w0, w1, w2));
}

fn sv_edge_signed(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  // Same as sv_edge but signed consistently with winding
  return (p.x - a.x)*(b.y - a.y) - (p.y - a.y)*(b.x - a.x);
}

fn sv_edge_len(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return max(length(b - a), 1e-6);
}

// Distance to the closest triangle edge in *pixels*
fn sv_min_edge_distance_px(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
  let e0 = abs(sv_edge_signed(p, a, b)) / sv_edge_len(a, b);
  let e1 = abs(sv_edge_signed(p, b, c)) / sv_edge_len(b, c);
  let e2 = abs(sv_edge_signed(p, c, a)) / sv_edge_len(c, a);
  return min(e0, min(e1, e2));
}

fn sv_tri_color(p: vec2<f32>, i0: u32, i1: u32, i2: u32) -> ColorHit {
  let a = verts.data[i0].pos;
  let b = verts.data[i1].pos;
  let c = verts.data[i2].pos;
  let bh = sv_tri_bary(p, a, b, c);
  if (!bh.hit) { return ColorHit(false, vec4<f32>(0.0), 0u, vec2<f32>(0.0)); }

  let w = bh.w;
  let uv = verts.data[i0].uv * w.x + verts.data[i1].uv * w.y + verts.data[i2].uv * w.z;
  var col = sv_sample(uv);
  if (col.a < 0.01) { return ColorHit(false, vec4<f32>(0.0), 0u, vec2<f32>(0.0)); }

  // --- Analytic edge AA ---
  let feather = 1.0;
  if (feather > 0.0) {
    let d = sv_min_edge_distance_px(p, a, b, c);  // pixels
    // Smooth coverage from 0..feather → 0..1 (softstep). Widen slightly for numeric stability.
    let cov = smoothstep(0.0, feather, d);
    // Multiply alpha by coverage (premultiplied not assumed here)
    col.a = col.a * cov;
  }

  // tri id is not known here; sv_shade_tile_pixel wraps this and sets it
  return ColorHit(true, col, 0u, uv);
}

fn sv_world_from_screen(pix: vec2<f32>) -> vec2<f32> {
  let invM = mat3x3<f32>(U.mat2d_inv_c0.xyz, U.mat2d_inv_c1.xyz, U.mat2d_inv_c2.xyz);
  let v = invM * vec3<f32>(pix, 1.0);
  return v.xy;
}

fn sv_shade_tile_pixel(p: vec2<f32>, px: u32, py: u32, tid: u32) -> ColorHit {
  let off = tile_offsets.data[tid];
  let cnt = tile_counts.data[tid];
  for (var k: u32 = 0u; k < cnt; k = k + 1u) {
    let t  = tile_tris.data[off + k];
    let i0 = indices.data[3u*t + 0u];
    let i1 = indices.data[3u*t + 1u];
    let i2 = indices.data[3u*t + 2u];
    let ch = sv_tri_color(p, i0, i1, i2);
    if (ch.hit) {
      return ColorHit(true, ch.color, t, ch.uv);
    }
  }
  return ColorHit(false, vec4<f32>(0.0), 0u, vec2<f32>(0.0));
}

// RNG Helper
fn wang_hash(x0: u32) -> u32 {
  var x = x0;
  x = (x ^ 61u) ^ (x >> 16u);
  x = x + (x << 3u);
  x = x ^ (x >> 4u);
  x = x * 0x27d4eb2du;
  x = x ^ (x >> 15u);
  return x;
}

// Combine pixel, frame, and any salt to make a good seed
fn sv_seed(px: u32, py: u32, salt: u32) -> u32 {
  return wang_hash(px ^ (py << 11u) ^ salt);
}

// Convert seed to [0,1)
fn sv_rand01(seed: u32) -> f32 {
  let h = wang_hash(seed);
  // 1/2^32 as f32
  return f32(h) * (1.0 / 4294967296.0);
}
