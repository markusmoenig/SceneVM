// 3D Header with utility functions. Cannot be replaced from the API

struct U3D {
  background: vec4<f32>,
  fb_size: vec2<u32>, _pad0: vec2<u32>,
  gp0: vec4<f32>, gp1: vec4<f32>, gp2: vec4<f32>,
  gp3: vec4<f32>, gp4: vec4<f32>, gp5: vec4<f32>,
  gp6: vec4<f32>, gp7: vec4<f32>, gp8: vec4<f32>, gp9: vec4<f32>,
  mat3d_c0: vec4<f32>,
  mat3d_c1: vec4<f32>,
  mat3d_c2: vec4<f32>,
  mat3d_c3: vec4<f32>,
  lights_count: u32,
  vm_flags: u32,
  anim_counter: u32,
  _pad_lights: u32,

  // --- Camera (matches Compute3DUniforms on the CPU) ---
  cam_pos:   vec4<f32>,  // xyz, pad
  cam_fwd:   vec4<f32>,  // xyz, pad
  cam_right: vec4<f32>,  // xyz, pad
  cam_up:    vec4<f32>,  // xyz, pad
  cam_vfov_deg:     f32, // perspective vertical FOV in degrees
  cam_ortho_half_h: f32, // ortho half-height
  cam_near:         f32,
  cam_far:          f32,
  cam_kind: u32,         // 0=OrthoIso, 1=OrbitPersp, 2=FirstPersonPersp
  _pad_cam: vec3<u32>,
};
@group(0) @binding(0) var<uniform> U: U3D;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var atlas_tex: texture_2d<f32>;
@group(0) @binding(3) var atlas_smp: sampler;

struct LightWGSL {
  header:   vec4<u32>,  // [light_type, emitting, _, _]
  position: vec4<f32>,  // xyz, _
  color:    vec4<f32>,  // rgb, _
  params0:  vec4<f32>,  // [intensity, radius, startD, endD]
  params1:  vec4<f32>,  // [flicker, _, _, _]
};
struct Lights { data: array<LightWGSL>, };
@group(0) @binding(4) var<storage, read> lights: Lights;

// Vert3D layout:
// - uv       : OBJECT UVs (not atlas-mapped). These can be any scale; we wrap in shader.
struct Vert3D {
  pos: vec3<f32>, _pad0: f32,
  uv: vec2<f32>,
  _pad_uv: vec2<f32>,
  tile_index: u32,
  _pad_tile: u32,
  _pad_tile2: vec2<f32>,
  normal: vec3<f32>, _pad2: f32
};
struct Verts3D { data: array<Vert3D> };
struct Indices { data: array<u32> };

@group(0) @binding(5) var<storage, read> verts3d: Verts3D;
@group(0) @binding(6) var<storage, read> indices3d: Indices;

// --- Scene-wide uniform grid (optional toggle via gp9.w) ---
struct Grid3DHeader {
  origin: vec4<f32>,     // xyz, pad
  cell_size: vec4<f32>,  // xyz, pad
  dims: vec4<u32>,       // nx, ny, nz, pad
  ranges: vec4<u32>,     // offsets_start, counts_start, tris_start, _
};
@group(0) @binding(7) var<uniform> gridH: Grid3DHeader;
struct GridDataBuffer {
  data: array<u32>,
};
@group(0) @binding(8)  var<storage, read> grid_data: GridDataBuffer;
@group(0) @binding(11) var atlas_mat_tex: texture_2d<f32>;
struct TileAnimMeta {
  first_frame: u32,
  frame_count: u32,
  _pad: vec2<u32>,
};
struct TileAnims {
  data: array<TileAnimMeta>,
};
struct TileFrame {
  ofs: vec2<f32>,
  scale: vec2<f32>,
};
struct TileFrames {
  data: array<TileFrame>,
};
@group(0) @binding(12) var<storage, read> tile_anims: TileAnims;
@group(0) @binding(13) var<storage, read> tile_frames: TileFrames;

fn sv_grid_active() -> bool { return U.gp9.w > 0.5; }

fn sv_write(px: u32, py: u32, c: vec4<f32>) {
  textureStore(color_out, vec2<i32>(i32(px), i32(py)), c);
}
fn sv_sample(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(atlas_tex, atlas_smp, uv, 0.0);
}
// ---- 3D utilities ----
// Full hit record including **geometric** normal (for debug/fallback). Shaders may
// still compute/interpolate their own shading normal using vertex data.
struct Hit3DFull { hit: bool, t: f32, u: f32, v: f32, Ng: vec3<f32> };

fn sv_ray_tri_full(ro: vec3<f32>, rd: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Hit3DFull {
  let e1 = b - a;
  let e2 = c - a;
  let p = cross(rd, e2);
  let det = dot(e1, p);
  if (abs(det) < 1e-8) { return Hit3DFull(false, 0.0, 0.0, 0.0, vec3<f32>(0.0)); }
  let inv_det = 1.0 / det;
  let tv = ro - a;
  let u = dot(tv, p) * inv_det;
  if (u < 0.0 || u > 1.0) { return Hit3DFull(false, 0.0, 0.0, 0.0, vec3<f32>(0.0)); }
  let q = cross(tv, e1);
  let v = dot(rd, q) * inv_det;
  if (v < 0.0 || u + v > 1.0) { return Hit3DFull(false, 0.0, 0.0, 0.0, vec3<f32>(0.0)); }
  let t = dot(e2, q) * inv_det;
  if (t <= 0.0) { return Hit3DFull(false, 0.0, 0.0, 0.0, vec3<f32>(0.0)); }
  // Geometric normal; flip to face the ray if needed for stability
  var Ng = normalize(cross(e1, e2));
  if (det > 0.0) { Ng = -Ng; }
  return Hit3DFull(true, t, u, v, Ng);
}

// --- Helpers: wrap UVs and clamp indices to valid SSBO ranges ---
fn clamp_index_u(i: u32, len: u32) -> u32 {
    return select(0u, min(i, max(len, 1u) - 1u), len > 0u);
}

// ---- UV wrapping helpers (GPU-side repeat inside atlas rect) ----
// OBJECT-UV bary mapping into atlas
fn sv_tile_frame(tile_index: u32) -> TileFrame {
  let meta_len = arrayLength(&tile_anims.data);
  if (meta_len == 0u) {
    return TileFrame(vec2<f32>(0.0), vec2<f32>(0.0));
  }
  let idx = min(tile_index, meta_len - 1u);
  let anim = tile_anims.data[idx];
  let count = max(anim.frame_count, 1u);
  let frames_len = arrayLength(&tile_frames.data);
  if (frames_len == 0u) {
    return TileFrame(vec2<f32>(0.0), vec2<f32>(0.0));
  }
  let frame_offset = anim.first_frame + (U.anim_counter % count);
  let frame_idx = min(frame_offset, frames_len - 1u);
  return tile_frames.data[frame_idx];
}

fn sv_tri_atlas_uv_obj(i0: u32, i1: u32, i2: u32, bu: f32, bv: f32) -> vec2<f32> {
  // Barycentric blend of per-vertex OBJECT uv
  let uv0 = verts3d.data[i0].uv;
  let uv1 = verts3d.data[i1].uv;
  let uv2 = verts3d.data[i2].uv;
  let w = 1.0 - bu - bv;
  let uv_obj = uv0 * w + uv1 * bu + uv2 * bv;

  let frame = sv_tile_frame(verts3d.data[i0].tile_index);

  // Repeat OBJECT uv, then scale into sub-rect and add offset
  var uv_wrapped = fract(uv_obj);          // [0,1) repeat in object space
  uv_wrapped.y = fract(1.0 - uv_wrapped.y); // flip Y so tiles aren't upside down
  let uv_atlas   = frame.ofs + uv_wrapped * frame.scale; // map into atlas sub-rect
  return uv_atlas;
}

// Sample atlas texture using barycentrics on a triangle with GPU-side repeat (object-UV based)
fn sv_tri_sample_albedo(i0: u32, i1: u32, i2: u32, bu: f32, bv: f32) -> vec4<f32> {
  let uv = sv_tri_atlas_uv_obj(i0, i1, i2, bu, bv);
  return sv_sample(uv);
}

fn sv_tri_sample_rmoe(i0: u32, i1: u32, i2: u32, bu: f32, bv: f32) -> vec4<f32> {
  let uv = sv_tri_atlas_uv_obj(i0, i1, i2, bu, bv);
  return textureSampleLevel(atlas_mat_tex, atlas_smp, uv, 0.0);
}

fn sv_interp3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, u: f32, v: f32) -> vec3<f32> {
  return a*(1.0-u-v) + b*u + c*v;
}

// ===== Uniform-grid DDA traversal over triangles =====

// Packed hit record returned by DDA tracing.
struct TraceHit {
  hit: bool,
  t: f32,        // distance along ray
  tri: u32,      // winning triangle index (in indices3d, tri = tix)
  u: f32,        // barycentric u
  v: f32,        // barycentric v
  Ng: vec3<f32>, // geometric normal
};

// Grid helpers
fn grid_bounds_min() -> vec3<f32> { return gridH.origin.xyz; }
fn grid_cell_size() -> vec3<f32> { return gridH.cell_size.xyz; }
fn grid_dims() -> vec3<u32> { return gridH.dims.xyz; }

fn grid_cell_index(ix: u32, iy: u32, iz: u32) -> u32 {
  let nx = gridH.dims.x; let ny = gridH.dims.y;
  return (iz * ny + iy) * nx + ix;
}

fn grid_world_to_cell(p: vec3<f32>) -> vec3<i32> {
  let minb = grid_bounds_min();
  let cs = grid_cell_size();
  let rel = (p - minb) / cs;
  return vec3<i32>(floor(rel));
}

fn clamp_cell(c: vec3<i32>) -> vec3<i32> {
  let d = grid_dims();
  return vec3<i32>(
    clamp(c.x, 0, i32(d.x) - 1),
    clamp(c.y, 0, i32(d.y) - 1),
    clamp(c.z, 0, i32(d.z) - 1)
  );
}

fn grid_bounds_max() -> vec3<f32> {
  return grid_bounds_min() + grid_cell_size() * vec3<f32>(grid_dims());
}

fn grid_offset(idx: u32) -> u32 {
  return grid_data.data[gridH.ranges.x + idx];
}

fn grid_count(idx: u32) -> u32 {
  return grid_data.data[gridH.ranges.y + idx];
}

fn grid_tri_index(idx: u32) -> u32 {
  return grid_data.data[gridH.ranges.z + idx];
}

// Ray/AABB for the whole grid, returns (hit, tEnter, tExit)
fn ray_box(ro: vec3<f32>, rd: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec3<f32> {
  let eps = 1e-6;

  // For each axis: if |rd| >= eps use rd, else use sign(rd)*eps (preserve sign)
  let rx = select(sign(rd.x) * eps, rd.x, abs(rd.x) >= eps);
  let ry = select(sign(rd.y) * eps, rd.y, abs(rd.y) >= eps);
  let rz = select(sign(rd.z) * eps, rd.z, abs(rd.z) >= eps);

  let inv = vec3<f32>(1.0 / rx, 1.0 / ry, 1.0 / rz);

  let t0 = (bmin - ro) * inv;
  let t1 = (bmax - ro) * inv;

  let tmin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
  let tmax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));

  let hit = select(0.0, 1.0, tmax >= max(tmin, 0.0));
  return vec3<f32>(hit, tmin, tmax);
}

struct DDAState {
  tMax:   vec3<f32>,  // absolute times when we hit the next boundary on each axis
  tDelta: vec3<f32>,  // absolute time increment to cross one full cell along each axis
  step:   vec3<i32>,  // -1 or +1 per axis
};

// p = point at tEnter (AABB entry), tEnter = absolute time
fn dda_setup(p: vec3<f32>, rd: vec3<f32>, cell: vec3<i32>, tEnter: f32) -> DDAState {
  let cs = grid_cell_size();
  let minb = grid_bounds_min();
  let fcell = vec3<f32>(cell);

  let step = vec3<i32>(
    select(-1, 1, rd.x >= 0.0),
    select(-1, 1, rd.y >= 0.0),
    select(-1, 1, rd.z >= 0.0)
  );

  // next boundary coordinate (on the side we are heading to)
  let nb_x = minb.x + (select(fcell.x, fcell.x + 1.0, step.x > 0) * cs.x);
  let nb_y = minb.y + (select(fcell.y, fcell.y + 1.0, step.y > 0) * cs.y);
  let nb_z = minb.z + (select(fcell.z, fcell.z + 1.0, step.z > 0) * cs.z);

  // safe reciprocals
  let inv = 1.0 / max(abs(rd), vec3<f32>(1e-32));

  // time from *p* to the next boundary per axis, then make them absolute by + tEnter
  let tMax = vec3<f32>(
    tEnter + (nb_x - p.x) * inv.x,
    tEnter + (nb_y - p.y) * inv.y,
    tEnter + (nb_z - p.z) * inv.z
  );

  // how much absolute time (Δt) to traverse exactly one cell per axis
  let tDelta = vec3<f32>(cs.x * inv.x, cs.y * inv.y, cs.z * inv.z);

  return DDAState(tMax, tDelta, step);
}

// Core: uniform-grid DDA traversal. Assumes grid buffers populated.
// tmin/tmax clip the segment (e.g., near/far planes).
fn sv_trace_grid(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32) -> TraceHit {
  // Intersect ray with grid AABB
  let bmin = grid_bounds_min();
  let bmax = grid_bounds_max();
  let rb = ray_box(ro, rd, bmin, bmax);
  if (rb.x < 0.5) { return TraceHit(false, 0.0, 0u, 0.0, 0.0, vec3<f32>(0.0)); }

  var tEnter = max(rb.y, tmin);
  let tExit = min(rb.z, tmax);
  if (tEnter > tExit) {
    return TraceHit(false, 0.0, 0u, 0.0, 0.0, vec3<f32>(0.0));
  }

  // Start cell: point at tEnter
  var p = ro + rd * tEnter;
  var cell = clamp_cell(grid_world_to_cell(p));

  // DDA setup
  var st = dda_setup(p, rd, cell, tEnter);
  var tNext = st.tMax;  // next boundary crossings

  let dims = grid_dims();

  var best_t = 1e30;
  var best_tri: u32 = 0u;
  var best_u = 0.0;
  var best_v = 0.0;
  var best_Ng = vec3<f32>(0.0);

  // Hard cap to prevent infinite loops; plenty for large screens
  let MAX_STEPS: u32 = 1u << 20u;
  var steps: u32 = 0u;

  loop {
    if (steps >= MAX_STEPS) { break; }
    steps = steps + 1u;

    // Cell index & triangle span
    let ix = u32(cell.x);
    let iy = u32(cell.y);
    let iz = u32(cell.z);
    let idx = grid_cell_index(ix, iy, iz);
    let off = grid_offset(idx);
    let cnt = grid_count(idx);

    // Test tris in this cell; accept the closest hit that lies BEFORE we cross the next cell boundary
    let tCellExit = min(tNext.x, min(tNext.y, tNext.z));
    for (var k: u32 = 0u; k < cnt; k = k + 1u) {
      let tri = grid_tri_index(off + k);

      // Fetch tri indices (tri references indices3d in packs of 3)
      let i0 = indices3d.data[3u*tri + 0u];
      let i1 = indices3d.data[3u*tri + 1u];
      let i2 = indices3d.data[3u*tri + 2u];

      // Triangle vertices
      let a = verts3d.data[i0].pos;
      let b = verts3d.data[i1].pos;
      let c = verts3d.data[i2].pos;

      let hit = sv_ray_tri_full(ro, rd, a, b, c);
      if (!hit.hit) { continue; }

      if (hit.t < best_t) {
        best_t  = hit.t;
        best_tri = tri;
        best_u  = hit.u;
        best_v  = hit.v;
        best_Ng = hit.Ng;
      }
    }

    if (best_t < 1e29) {
      // Hit inside current cell slab -> earliest along ray
      return TraceHit(true, best_t, best_tri, best_u, best_v, best_Ng);
    }

    // Advance to next cell along the smallest tNext
    if (tNext.x < tNext.y) {
      if (tNext.x < tNext.z) {
        cell.x += st.step.x;
        tEnter = tNext.x;
        tNext.x += abs(st.tDelta.x);
      } else {
        cell.z += st.step.z;
        tEnter = tNext.z;
        tNext.z += abs(st.tDelta.z);
      }
    } else {
      if (tNext.y < tNext.z) {
        cell.y += st.step.y;
        tEnter = tNext.y;
        tNext.y += abs(st.tDelta.y);
      } else {
        cell.z += st.step.z;
        tEnter = tNext.z;
        tNext.z += abs(st.tDelta.z);
      }
    }

    // Out of grid or beyond segment
    if (tEnter > tExit) { break; }
    if (cell.x < 0 || cell.y < 0 || cell.z < 0) { break; }
    if (u32(cell.x) >= dims.x || u32(cell.y) >= dims.y || u32(cell.z) >= dims.z) { break; }
  }

  return TraceHit(false, 0.0, 0u, 0.0, 0.0, vec3<f32>(0.0));
}
// ===== end DDA =====

// TBN from triangle positions and OBJECT-space UVs (no atlas mapping here),
// using geometric normal for stability.
fn sv_tri_tbn(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>,
              uv0: vec2<f32>, uv1: vec2<f32>, uv2: vec2<f32>) -> mat3x3<f32> {
  let e1 = b - a;
  let e2 = c - a;
  let d1 = uv1 - uv0;
  let d2 = uv2 - uv0;
  let r = 1.0 / max(d1.x * d2.y - d1.y * d2.x, 1e-8);

  var T = normalize((e1 * d2.y - e2 * d1.y) * r);
  var Ng = normalize(cross(e1, e2));
  T = normalize(T - Ng * dot(Ng, T));
  let B = normalize(cross(Ng, T));
  return mat3x3<f32>(T, B, Ng);
}

// Luma from color (height proxy)
fn sv_luma(rgb: vec3<f32>) -> f32 {
  return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

// Camera
struct Ray { ro: vec3<f32>, rd: vec3<f32> };

fn cam_ray(uv: vec2<f32>) -> Ray {
  // uv in [0,1]^2 pixel center; build NDC in [-1,1]
  let res = vec2<f32>(f32(U.fb_size.x), f32(U.fb_size.y));
  let ndc = (uv * 2.0 - vec2<f32>(1.0,1.0)) * vec2<f32>(1.0, -1.0); // y up

  if (U.cam_kind == 0u) {
    // Ortho: origin scans a rectangle on a plane; direction = forward
    let aspect = res.x / max(res.y, 1.0);
    let half_w = U.cam_ortho_half_h * aspect;
    let p = U.cam_pos.xyz
          + U.cam_right.xyz * (ndc.x * half_w)
          + U.cam_up.xyz    * (ndc.y * U.cam_ortho_half_h);
    return Ray(p, normalize(U.cam_fwd.xyz));
  } else {
    // Perspective: pinhole at cam_pos
    // screen basis on near plane using vfov
    let tan_half = tan(radians(U.cam_vfov_deg) * 0.5);
    let aspect = res.x / max(res.y, 1.0);
    let dx = ndc.x * aspect * tan_half;
    let dy = ndc.y * tan_half;
    let dir = normalize(U.cam_fwd.xyz + U.cam_right.xyz * dx + U.cam_up.xyz * dy);
    return Ray(U.cam_pos.xyz, dir);
  }
}
