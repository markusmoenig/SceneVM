// Non-empty dummy buffers for wgpu STORAGE bindings when a scene grid is empty.
const DUMMY_U32_1: [u32; 1] = [0];
const DUMMY_I32_1: [i32; 1] = [0];

use crate::{Chunk, Light, LightType, Material, Texture};
use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use uuid::Uuid;
use vek::{Mat3, Mat4, Vec3, Vec4};

// --- Scene-wide acceleration structures (uniform grid over all 3D geometry) ---
#[derive(Debug, Clone, Default)]
pub struct SceneGridAccel {
    pub origin: vek::Vec3<f32>,    // world-space min of the grid AABB
    pub cell_size: vek::Vec3<f32>, // world size of a cell (x,y,z)
    pub dims: [u32; 3],            // nx, ny, nz
    // CSR arrays for cell -> tri list
    pub cell_offsets: Vec<u32>, // len = nx*ny*nz
    pub cell_counts: Vec<u32>,  // len = nx*ny*nz
    pub cell_tris: Vec<u32>,    // flattened triangle indices
    // Per-triangle metadata (kept flat and aligned with scene's 3D tri order)
    pub tri_tile: Vec<u32>,  // tri -> tile index (for sampling)
    pub tri_layer: Vec<i32>, // tri -> layer (optional ordering/debug)
}

#[derive(Debug, Clone, Default)]
pub struct SceneAccel {
    pub grid: SceneGridAccel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// The Geometry Identifier for polygons and triangles.
pub enum GeoId {
    Unknown(u32),
    Vertex(u32),
    Linedef(u32),
    Sector(u32),
    Character(u32),
    Item(u32),
    Light(u32),
    ItemLight(u32),
    Triangle(u32),
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vert2DPod {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vert3DPod {
    // 0..12
    pub pos: [f32; 3],
    // pad to 16 so next member is 16-aligned
    pub _pad_pos: f32,
    // 16..23
    pub uv: [f32; 2],
    // pad to 32 so next member starts at 32
    pub _pad_uv: [f32; 2],
    // 32..44
    pub normal: [f32; 3],
    // pad to 48 for 16B alignment of array stride
    pub _pad_n: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightPod {
    // header: [light_type, emitting, pad, pad]
    pub header: [u32; 4],
    pub position: [f32; 4], // xyz, pad
    pub color: [f32; 4],    // rgb, pad
    // params0: [intensity, radius, start_distance, end_distance]
    pub params0: [f32; 4],
    // params1: [flicker, pad, pad, pad]
    pub params1: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialPod {
    pub tint: [f32; 4],
    pub rmoe: [f32; 4],
    pub model: [f32; 4],
}

/// VM instruction set
#[derive(Debug)]
pub enum Atom {
    /// Add a tile with `id`, dimensions, and animation frames (RGBA8). Each frame is tightly packed width*height*4 bytes.
    AddTile {
        id: Uuid,
        width: u32,
        height: u32,
        frames: Vec<Vec<u8>>, // frames[f][row*width*4 .. (row+1)*width*4]
    },
    /// Add a solid-color 1x1 tile with `id` and RGBA color.
    AddSolid {
        id: Uuid,
        color: [u8; 4],
    },
    /// Build the atlas for all frames
    BuildAtlas,
    /// Add a polygon (world coords) that references a tile by UUID into the CURRENT chunk; indices are local to the chunk.
    AddPoly {
        id: GeoId,     // geometry id (stable within the chunk)
        tile_id: Uuid, // which tile's frames to sample from
        vertices: Vec<[f32; 2]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<(usize, usize, usize)>,
        material_id: Option<Uuid>,
    },
    /// Add a 3D polygon (world coords) that references a tile by UUID; indices are local to the chunk.
    AddPoly3D {
        id: GeoId,
        tile_id: Uuid,
        vertices: Vec<[f32; 4]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<(usize, usize, usize)>,
        material_id: Option<Uuid>,
    },
    /// Add a simple 2D line strip as thick segments tessellated into quads (no caps/joins)
    /// Points are in world coordinates; width is in the same units.
    AddLineStrip2D {
        id: GeoId,
        tile_id: Uuid,
        points: Vec<[f32; 2]>,
        width: f32,
        material_id: Option<Uuid>,
    },
    /// Create an empty chunk (no switch)
    NewChunk {
        id: Uuid,
    },
    /// Insert or replace an entire chunk in one go (prepared externally, e.g., in Rusterix)
    AddChunk {
        id: Uuid,
        chunk: Chunk,
    },
    /// Remove an existing chunk; if it is the current chunk, unset it
    RemoveChunk {
        id: Uuid,
    },
    /// Remove a chunk at a given origin (in chunk grid coordinates)
    RemoveChunkAt {
        origin: vek::Vec2<i32>,
    },
    /// Switch the current chunk (created if missing)
    SetCurrentChunk {
        id: Uuid,
    },
    /// Set the current animation counter (frame index modulo each tile's frame count)
    SetAnimationCounter(usize),
    /// Set background color/params shared by 2D and 3D
    SetBackground(Vec4<f32>),
    /// General-purpose vec4 slots shared by 2D and 3D
    SetGP0(Vec4<f32>),
    SetGP1(Vec4<f32>),
    SetGP2(Vec4<f32>),
    SetGP3(Vec4<f32>),
    SetGP4(Vec4<f32>),
    SetGP5(Vec4<f32>),
    SetGP6(Vec4<f32>),
    SetGP7(Vec4<f32>),
    SetGP8(Vec4<f32>),
    SetGP9(Vec4<f32>),
    /// Switch between 2D and 3D compute drawing
    SetRenderMode(RenderMode),
    /// Set a 2D transform (Mat3) applied on CPU to polygon vertices before 2D compute draw
    SetTransform2D(Mat3<f32>),
    /// Set a 3D transform (Mat4) applied on CPU to polygon vertices before 3D compute draw
    SetTransform3D(Mat4<f32>),
    /// Set current 2D/3D layer for subsequently added geometry
    SetLayer(i32),
    /// Toggle visibility for a specific geometry id across all chunks
    SetGeoVisible {
        id: GeoId,
        visible: bool,
    },
    /// Provide a custom WGSL body for the 2D compute shader. The VM will prepend a header and compile at runtime.
    SetSource2D(String),
    /// Provide a custom WGSL body for the 3D compute shader. The VM will prepend a header and compile at runtime.
    SetSource3D(String),
    /// Clear EVERYTHING: tiles, atlas, scene (chunks), counters and modes
    Clear,
    /// Clear only the tiles and atlas (keep scene/chunks intact)
    ClearTiles,
    /// Clear only the scene geometry (chunks & current selection), keep tiles/atlas intact
    ClearGeometry,
    /// Add a light to the scene
    AddLight {
        id: GeoId,
        light: Light,
    },
    /// Remove a light by its id
    RemoveLight {
        id: GeoId,
    },
    /// Remove all lights from the scene
    ClearLights,
    /// Add/replace a material
    AddMaterial {
        id: Uuid,
        material: Material,
    },
    /// Remove a material
    RemoveMaterial {
        id: Uuid,
    },
    /// Remove all materials
    ClearMaterials,
    /// Assign a material to a specific geometry id (2D or 3D)
    SetGeoMaterial {
        id: GeoId,
        material_id: Option<Uuid>,
    },
    /// Build/replace the global scene uniform grid over all current 3D geometry
    SetSceneGridCells {
        target_cells: u32,
    },
    /// Reset the scene acceleration structure (will be rebuilt on next BuildSceneGrid)
    ClearSceneGrid,
}

#[derive(Debug, Clone)]
pub struct AtlasEntry {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Clone)]
pub struct Poly2D {
    pub id: GeoId,
    pub tile_id: Uuid,
    pub vertices: Vec<[f32; 2]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<(usize, usize, usize)>, // triangle list, LOCAL to its chunk
    pub transform: Mat3<f32>,                // per-poly local transform
    pub layer: i32,                          // visual layer; higher draws on top
    pub visible: bool,                       // if false, skipped during draw
    pub material_id: Option<Uuid>,
}

#[derive(Debug, Clone)]
pub struct Poly3D {
    pub id: GeoId,
    pub tile_id: uuid::Uuid,
    pub vertices: Vec<[f32; 4]>, // world-space XYZ(W)
    pub uvs: Vec<[f32; 2]>,      // per-vertex UV
    pub indices: Vec<(usize, usize, usize)>,
    pub layer: i32, // for future (not used by ray depth)
    pub visible: bool,
    pub material_id: Option<Uuid>,
}

#[derive(Debug)]
struct Tile {
    w: u32,
    h: u32,
    frames: Vec<Vec<u8>>,
}

// GPU rendering resources managed directly by VM
pub struct VMGpu {
    pub pipeline_2d: wgpu::RenderPipeline,
    pub globals_buf: wgpu::Buffer,
    pub globals_bgl: wgpu::BindGroupLayout,
    pub atlas_bgl: wgpu::BindGroupLayout,
    pub globals_bg: Option<wgpu::BindGroup>,
    pub atlas_bg: Option<wgpu::BindGroup>,
    pub vbuf: Option<wgpu::Buffer>,
    pub ibuf: Option<wgpu::Buffer>,
    pub index_count: u32,
    pub sampler: wgpu::Sampler,
    // --- Compute pipelines and uniforms (lazily created)
    pub compute2d_pipeline: Option<wgpu::ComputePipeline>,
    pub compute3d_pipeline: Option<wgpu::ComputePipeline>,
    pub u2d_buf: Option<wgpu::Buffer>,
    pub u3d_buf: Option<wgpu::Buffer>,
    pub u2d_bgl: Option<wgpu::BindGroupLayout>,
    pub u3d_bgl: Option<wgpu::BindGroupLayout>,
    pub u2d_bg: Option<wgpu::BindGroup>,
    pub u3d_bg: Option<wgpu::BindGroup>,
    pub v2d_ssbo: Option<wgpu::Buffer>,
    pub i2d_ssbo: Option<wgpu::Buffer>,
    pub v3d_ssbo: Option<wgpu::Buffer>,
    pub i3d_ssbo: Option<wgpu::Buffer>,
    // --- Tiling
    pub tile_offsets: Option<wgpu::Buffer>,
    pub tile_counts: Option<wgpu::Buffer>,
    pub tile_tris: Option<wgpu::Buffer>,
    // Lights
    pub lights_ssbo: Option<wgpu::Buffer>,
    // --- Scene-wide uniform grid buffers (3D)
    pub grid_hdr: Option<wgpu::Buffer>,
    pub grid_offsets: Option<wgpu::Buffer>,
    pub grid_counts: Option<wgpu::Buffer>,
    pub grid_tris: Option<wgpu::Buffer>,
    pub tri_tile: Option<wgpu::Buffer>,
    pub tri_layer: Option<wgpu::Buffer>,
    // Materials
    pub tri_mat2d: Option<wgpu::Buffer>,
    pub tri_mat3d: Option<wgpu::Buffer>,
    pub materials_ssbo: Option<wgpu::Buffer>,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    pub tx: f32,
    pub ty: f32,
    pub scale: f32,
    _pad0: f32,
    pub atlas_w: f32,
    pub atlas_h: f32,
    _pad1: f32,
    _pad2: f32,
}

pub const SCENEVM_2D_WGSL: &str = r#"
struct Globals {
  tx: f32, ty: f32, scale: f32, _pad0: f32,
  atlas_w: f32, atlas_h: f32, _pad1: f32, _pad2: f32,
};
@group(0) @binding(0) var<uniform> G: Globals;
@group(1) @binding(0) var atlas_tex: texture_2d<f32>;
@group(1) @binding(1) var atlas_smp: sampler;

struct VsIn { @location(0) pos: vec2<f32>, @location(1) uv: vec2<f32> };
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs_main(in: VsIn) -> VsOut {
  var out: VsOut;
  // Temporary mapping: interpret pos as pixels in an atlas-sized viewport
  let x = (in.pos.x / G.atlas_w) * 2.0 - 1.0;
  let y = (in.pos.y / G.atlas_h) * -2.0 + 1.0;
  out.pos = vec4<f32>(x, y, 0.0, 1.0);
  out.uv = in.uv;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  return textureSample(atlas_tex, atlas_smp, in.uv);
}
"#;

// --- Compute pipeline uniforms and WGSL shaders ---
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Compute2DUniforms {
    pub background: [f32; 4], // was param
    pub fb_size: [u32; 2],
    _pad0: [u32; 2],
    pub gp0: [f32; 4], // general-purpose vec4s
    pub gp1: [f32; 4],
    pub gp2: [f32; 4],
    pub gp3: [f32; 4],
    pub gp4: [f32; 4],
    pub gp5: [f32; 4],
    pub gp6: [f32; 4],
    pub gp7: [f32; 4],
    pub gp8: [f32; 4],
    pub gp9: [f32; 4],
    // Mat3<f32> as 3 padded vec4 columns (col-major), .w is padding
    pub mat2d_c0: [f32; 4],
    pub mat2d_c1: [f32; 4],
    pub mat2d_c2: [f32; 4],
    // Inverse 2D matrix columns
    pub mat2d_inv_c0: [f32; 4],
    pub mat2d_inv_c1: [f32; 4],
    pub mat2d_inv_c2: [f32; 4],

    pub lights_count: u32,
    // WGSL: a vec3<u32> after a u32 costs 12B pad + 16B vec3 slot = 28B total.
    _pad_lights_align: [u32; 3],
    _pad_lights_vec3: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Compute3DUniforms {
    pub background: [f32; 4],
    pub fb_size: [u32; 2],
    _pad0: [u32; 2],
    pub gp0: [f32; 4],
    pub gp1: [f32; 4],
    pub gp2: [f32; 4],
    pub gp3: [f32; 4],
    pub gp4: [f32; 4],
    pub gp5: [f32; 4],
    pub gp6: [f32; 4],
    pub gp7: [f32; 4],
    pub gp8: [f32; 4],
    pub gp9: [f32; 4],
    // Mat4<f32> as 4 vec4 columns (col-major)
    pub mat3d_c0: [f32; 4],
    pub mat3d_c1: [f32; 4],
    pub mat3d_c2: [f32; 4],
    pub mat3d_c3: [f32; 4],

    pub lights_count: u32,
    _pad_lights_align: [u32; 3],
    _pad_lights_vec3: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Grid3DHeader {
    pub origin: [f32; 4],    // xyz, pad
    pub cell_size: [f32; 4], // xyz, pad
    pub dims: [u32; 4],      // nx, ny, nz, pad
}

pub const SCENEVM_2D_CS_WGSL: &str = r#"
struct U2D { background: vec4<f32>, fb_size: vec2<u32>, _pad: vec2<u32> };
@group(0) @binding(0) var<uniform> U: U2D;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= U.fb_size.x || gid.y >= U.fb_size.y) { return; }
  let uv = vec2<f32>(f32(gid.x)/f32(U.fb_size.x), f32(gid.y)/f32(U.fb_size.y));
  // For now: solid color with simple uv tint; later: raster & lighting
  let col = /*vec4<f32>(U.background.xyz, 1.0); */ vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  textureStore(color_out, vec2<i32>(i32(gid.x), i32(gid.y)), col);
}
"#;

pub const SCENEVM_3D_CS_WGSL: &str = r#"
struct U3D { background: vec4<f32>, fb_size: vec2<u32>, _pad: vec2<u32>, };
@group(0) @binding(0) var<uniform> U: U3D;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= U.fb_size.x || gid.y >= U.fb_size.y) { return; }
  // Placeholder: gradient with background.x as brightness; later we pathtrace here
  let uv = vec2<f32>(f32(gid.x)/f32(U.fb_size.x), f32(gid.y)/f32(U.fb_size.y));
  let b = U.background.x;
  let col = vec4<f32>(uv.x*b, uv.y*b, b, 1.0);
  textureStore(color_out, vec2<i32>(i32(gid.x), i32(gid.y)), col);
}
"#;

pub const SCENEVM_2D_HEADER: &str = r#"
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

@group(0) @binding(10) var<storage, read> tri_mat2d: U32s;

struct MaterialWGSL {
  tint:  vec4<f32>,
  rmoe:  vec4<f32>,
  model: vec4<f32>,
};
struct Materials { data: array<MaterialWGSL>, };
@group(0) @binding(11) var<storage, read> materials: Materials;

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
struct ColorHit { hit: bool, color: vec4<f32>, tri: u32 };

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
  if (!bh.hit) { return ColorHit(false, vec4<f32>(0.0), 0u); }

  let w = bh.w;
  let uv = verts.data[i0].uv * w.x + verts.data[i1].uv * w.y + verts.data[i2].uv * w.z;
  var col = sv_sample(uv);
  if (col.a < 0.01) { return ColorHit(false, vec4<f32>(0.0), 0u); }

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
  return ColorHit(true, col, 0u);
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
      return ColorHit(true, ch.color, t);
    }
  }
  return ColorHit(false, vec4<f32>(0.0), 0u);
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
// ----- end helpers -----
"#;

pub const SCENEVM_3D_HEADER: &str = r#"
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
  _pad_lights: vec3<u32>,
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

// Geometry
struct Vert3D { pos: vec3<f32>, _pad0: f32, uv: vec2<f32>, _pad1: vec2<f32>, normal: vec3<f32>, _pad2: f32 };
struct Verts3D { data: array<Vert3D> };
struct Indices { data: array<u32> };

@group(0) @binding(5) var<storage, read> verts3d: Verts3D;
@group(0) @binding(6) var<storage, read> indices3d: Indices;

// --- Scene-wide uniform grid (optional toggle via gp9.w) ---
struct Grid3DHeader {
  origin: vec4<f32>,     // xyz, pad
  cell_size: vec4<f32>,  // xyz, pad
  dims: vec4<u32>,       // nx, ny, nz, pad
};
@group(0) @binding(7) var<uniform> gridH: Grid3DHeader;
struct U32s { data: array<u32> };
struct I32s { data: array<i32> };

@group(0) @binding(8)  var<storage, read> grid_offsets: U32s;
@group(0) @binding(9)  var<storage, read> grid_counts:  U32s;
@group(0) @binding(10) var<storage, read> grid_tris:    U32s;
// material index per triangle (aligned with indices3d triangles)
@group(0) @binding(13) var<storage, read> tri_mat: U32s;

// Materials array
struct MaterialWGSL {
  tint:  vec4<f32>,
  rmoe:  vec4<f32>,
  model: vec4<f32>,
};
struct Materials { data: array<MaterialWGSL>, };
@group(0) @binding(14) var<storage, read> materials: Materials;

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
    let off = grid_offsets.data[idx];
    let cnt = grid_counts.data[idx];

    // Test tris in this cell; accept the closest hit that lies BEFORE we cross the next cell boundary
    let tCellExit = min(tNext.x, min(tNext.y, tNext.z));
    for (var k: u32 = 0u; k < cnt; k = k + 1u) {
      let tri = grid_tris.data[off + k];

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

// TBN from triangle positions/uvs, using geometric normal for stability.
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
"#;

pub const DEFAULT_2D_BODY: &str = r#"
@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= U.fb_size.x || py >= U.fb_size.y) { return; }

  // Clear to background first
  sv_write(px, py, U.background);

  let p = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);
  let tid = tile_of_px(px, py);
  let ch = sv_shade_tile_pixel(p, px, py, tid);
  if (ch.hit) {
    // Material look-up for winning triangle
    let m_idx = tri_mat2d.data[ch.tri];
    let M = materials.data[m_idx];

    // Base texture color → apply tint & opacity; add emission (simple)
    let base = ch.color;
    let rgb = base.xyz * M.tint.xyz + M.rmoe.w * M.tint.xyz; // tint + emission
    let a   = base.a * M.rmoe.z;                             // opacity
    sv_write(px, py, vec4<f32>(rgb, a));
  }
}
"#;

pub const DEFAULT_3D_BODY: &str = r#"
  // --- Test Lambert shading (kept in BODY so headers stay generic) ---
  fn lambert_pointlights(P: vec3<f32>, N: vec3<f32>, base_col: vec3<f32>) -> vec3<f32> {
    var diffuse = vec3<f32>(0.0);
    // Use background as ambient; make it visible out of the box
    let ambient = U.background.xyz;

    for (var li: u32 = 0u; li < U.lights_count; li = li + 1u) {
      if (lights.data[li].header.y == 0u) { continue; } // emitting flag

      let Lp = lights.data[li].position;
      let Lc = lights.data[li].color.xyz;
      let Li = lights.data[li].params0.x + lights.data[li].params1.x;   // intensity + flicker

      let start_d = lights.data[li].params0.z;
      let end_d   = max(lights.data[li].params0.w, start_d + 1e-3);
      let L = Lp.xyz - P;
      let dist2 = max(dot(L, L), 1e-6);
      let dist = sqrt(dist2);
      let Ldir = normalize(L);

      // Always two-sided: use |N·L|
      let ndotl = abs(dot(N, Ldir));

      let fall = clamp((end_d - dist) / max(end_d - start_d, 1e-3), 0.0, 1.0);
      let atten = Li * ndotl * fall / dist2;
      diffuse += Lc * atten;
    }
    return base_col * (ambient + diffuse);
  }
  
@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x; let py = gid.y;
  if (px >= U.fb_size.x || py >= U.fb_size.y) { return; }

  // Unpack camera (unchanged) ...
  let cam_pos = U.gp0.xyz;
  let fovy = U.gp0.w;
  let dir = normalize(U.gp1.xyz);
  let aspect = U.gp1.w;
  let right = normalize(U.gp2.xyz);
  let up = normalize(U.gp3.xyz);

  let sx = (f32(px) + 0.5) / f32(U.fb_size.x);
  let sy = (f32(py) + 0.5) / f32(U.fb_size.y);
  let x_ndc = 2.0 * sx - 1.0;
  let y_ndc = 1.0 - 2.0 * sy;
  let tproj = tan(0.5 * fovy);
  let rd = normalize(dir + x_ndc * tproj * aspect * right + y_ndc * tproj * up);
  let ro = cam_pos;

  // ===== choose tracing mode =====
  var hit_any = false;
  var best_t  = 1e30;
  var best_tri: u32 = 0u;
  var best_u = 0.0;
  var best_v = 0.0;

  if (sv_grid_active()) {
    let th = sv_trace_grid(ro, rd, 0.001, 1e6);
    if (th.hit) {
      hit_any = true;
      best_t = th.t;
      best_tri = th.tri;
      best_u = th.u;
      best_v = th.v;
    }
  } else {
    // Brute-force: loop all triangles in indices3d
    let tri_count: u32 = arrayLength(&indices3d.data) / 3u;
    for (var tri: u32 = 0u; tri < tri_count; tri = tri + 1u) {
      let i0 = indices3d.data[3u*tri + 0u];
      let i1 = indices3d.data[3u*tri + 1u];
      let i2 = indices3d.data[3u*tri + 2u];
      let a = verts3d.data[i0].pos;
      let b = verts3d.data[i1].pos;
      let c = verts3d.data[i2].pos;
      let h = sv_ray_tri_full(ro, rd, a, b, c);
      if (h.hit && h.t < best_t) {
        hit_any = true;
        best_t = h.t;
        best_tri = tri;
        best_u = h.u;
        best_v = h.v;
      }
    }
  }

  if (!hit_any) {
    sv_write(px, py, U.background);
    return;
  }

  // Interpolate UV & smooth normal
  let i0 = indices3d.data[3u*best_tri + 0u];
  let i1 = indices3d.data[3u*best_tri + 1u];
  let i2 = indices3d.data[3u*best_tri + 2u];

  let uv0 = verts3d.data[i0].uv; let n0 = verts3d.data[i0].normal;
  let uv1 = verts3d.data[i1].uv; let n1 = verts3d.data[i1].normal;
  let uv2 = verts3d.data[i2].uv; let n2 = verts3d.data[i2].normal;

  let w0 = 1.0 - best_u - best_v;
  let uv = uv0*w0 + uv1*best_u + uv2*best_v;
  var N = normalize(n0*w0 + n1*best_u + n2*best_v);

  // when filling Compute3DUniforms u:
  // self.gp8.x = 1.0; // bump strength (0 = off)
  // self.gp9.x = 1.0 / (self.atlas.width as f32);
  // self.gp9.y = 1.0 / (self.atlas.height as f32);

  // Optional bump from the polygon's own texture as height
  if (U.gp8.x > 0.0) {
    // Reconstruct triangle positions for TBN
    let a = verts3d.data[i0].pos;
    let b = verts3d.data[i1].pos;
    let c = verts3d.data[i2].pos;

    // 1 texel steps in atlas UV space (provided by CPU)
    let du = vec2<f32>(U.gp9.x, 0.0);
    let dv = vec2<f32>(0.0, U.gp9.y);

    // Sample height at uv and neighbors (use color as height proxy)
    let h  = sv_luma(textureSampleLevel(atlas_tex, atlas_smp, uv, 0.0).xyz);
    let hx = sv_luma(textureSampleLevel(atlas_tex, atlas_smp, uv + du, 0.0).xyz);
    let hy = sv_luma(textureSampleLevel(atlas_tex, atlas_smp, uv + dv, 0.0).xyz);

    // Finite differences
    let dhdu = (hx - h);
    let dhdv = (hy - h);

    // Tangent frame of the triangle
    let TBN = sv_tri_tbn(a, b, c, uv0, uv1, uv2);

    // Map height gradient into tangent space normal and to world space
    let n_ts = normalize(vec3<f32>(-dhdu * U.gp8.x, -dhdv * U.gp8.x, 1.0));
    let n_ws = normalize(TBN * n_ts);

    // Blend with your smooth vertex normal for stability
    N = normalize(mix(N, n_ws, clamp(U.gp8.x, 0.0, 1.0)));
  }  

  let P = ro + rd * best_t;

    let base_col = textureSampleLevel(atlas_tex, atlas_smp, uv, 0.0);
    if (dot(N, rd) > 0.0) { N = -N; } // two-sided

    // Material lookup for the winning triangle
    let m_idx = tri_mat.data[best_tri];
    let M = materials.data[m_idx];
    let base_rgb = base_col.xyz * M.tint.xyz;

    let lit = lambert_pointlights(P, N, base_rgb);
    // Add simple emission, apply opacity (from material)
    let final_rgb = lit + M.rmoe.w * M.tint.xyz;
    let final_a = base_col.a * M.rmoe.z;
    sv_write(px, py, vec4<f32>(final_rgb, final_a));
}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Compute2D,
    Compute3D,
}

/// A tiny, CPU-side VM that collects tiles and builds a texture atlas.
/// Packing strategy: simple shelf packer (rows), stable order by insertion.
pub struct VM {
    tiles_map: FxHashMap<Uuid, Tile>,
    tiles_order: Vec<Uuid>, // insertion order for stable packing
    pub atlas: Texture,     // CPU/GPU-capable atlas texture
    pub atlas_map: FxHashMap<Uuid, Vec<AtlasEntry>>, // per-tile frame rects in atlas order

    // Scene content grouped into chunks (for streaming/load-save). Indices are local per-chunk.
    pub chunks_map: FxHashMap<Uuid, Chunk>,
    pub current_chunk: Option<Uuid>,

    pub animation_counter: usize,
    pub render_mode: RenderMode,

    pub gpu: Option<VMGpu>,
    // --- Compute pipeline params (shared by 2D/3D)
    pub background: Vec4<f32>,
    pub gp0: Vec4<f32>,
    pub gp1: Vec4<f32>,
    pub gp2: Vec4<f32>,
    pub gp3: Vec4<f32>,
    pub gp4: Vec4<f32>,
    pub gp5: Vec4<f32>,
    pub gp6: Vec4<f32>,
    pub gp7: Vec4<f32>,
    pub gp8: Vec4<f32>,
    pub gp9: Vec4<f32>,
    // --- Programmable compute shader sources
    pub source2d: String,
    pub source3d: String,

    pub transform2d: Mat3<f32>,
    pub transform3d: Mat4<f32>,

    pub lights: FxHashMap<GeoId, Light>,
    pub materials: FxHashMap<Uuid, Material>,

    pub current_layer: i32,

    // Scene-wide 3D acceleration via grid
    pub scene_grid_cells: u32,
    pub scene_accel: SceneAccel,
    pub accel_dirty: bool,
}

impl VM {
    /// Create a VM with a fixed-size atlas (atlas_w x atlas_h).
    pub fn new(atlas_w: u32, atlas_h: u32) -> Self {
        Self {
            tiles_map: FxHashMap::default(),
            tiles_order: Vec::new(),
            atlas: Texture::new(atlas_w, atlas_h),
            atlas_map: FxHashMap::default(),
            chunks_map: FxHashMap::default(),
            current_chunk: None,
            animation_counter: 0,
            render_mode: RenderMode::Compute2D,
            gpu: None,
            background: Vec4::new(1.0, 0.8, 0.2, 1.0),
            gp0: Vec4::new(0.0, 0.0, 0.0, 0.0),
            gp1: Vec4::new(0.0, 0.0, 0.0, 0.0),
            gp2: Vec4::new(0.0, 0.0, 0.0, 0.0),
            gp3: Vec4::new(0.0, 0.0, 0.0, 0.0),
            gp4: Vec4::new(0.0, 0.0, 0.0, 0.0),
            gp5: Vec4::new(0.0, 0.0, 0.0, 0.0),
            gp6: Vec4::new(0.0, 0.0, 0.0, 0.0),
            gp7: Vec4::new(0.0, 0.0, 0.0, 0.0),
            gp8: Vec4::new(0.0, 0.0, 0.0, 0.0),
            gp9: Vec4::new(0.0, 0.0, 0.0, 0.0),
            source2d: DEFAULT_2D_BODY.to_string(),
            source3d: DEFAULT_3D_BODY.to_string(),
            transform2d: Mat3::identity(),
            transform3d: Mat4::identity(),
            lights: FxHashMap::default(),
            materials: FxHashMap::default(),
            current_layer: 0,
            scene_accel: SceneAccel::default(),
            accel_dirty: true,
            scene_grid_cells: 5000,
        }
    }

    /// Interpret one instruction.
    pub fn execute(&mut self, atom: Atom) {
        match atom {
            Atom::SetGeoVisible { id, visible } => {
                for ch in self.chunks_map.values_mut() {
                    if let Some(p) = ch.polys_map.get_mut(&id) {
                        p.visible = visible;
                    }
                    if let Some(p3) = ch.polys3d_map.get_mut(&id) {
                        p3.visible = visible;
                        self.accel_dirty = true;
                    }
                }
            }
            Atom::AddTile {
                id,
                width,
                height,
                frames,
            } => {
                // Basic validation: ensure each frame has enough bytes; pad/trim as needed
                let need = (width as usize) * (height as usize) * 4;
                let frames: Vec<Vec<u8>> = frames
                    .into_iter()
                    .map(|mut f| {
                        if f.len() < need {
                            f.resize(need, 0);
                        }
                        if f.len() > need {
                            f.truncate(need);
                        }
                        f
                    })
                    .collect();
                let is_new = !self.tiles_map.contains_key(&id);
                self.tiles_map.insert(
                    id,
                    Tile {
                        w: width,
                        h: height,
                        frames,
                    },
                );
                if is_new {
                    self.tiles_order.push(id);
                }
            }
            Atom::AddSolid { id, color } => {
                // Create a 1x1 tile with a single frame of the given color
                let frame = color.to_vec();
                let is_new = !self.tiles_map.contains_key(&id);
                self.tiles_map.insert(
                    id,
                    Tile {
                        w: 1,
                        h: 1,
                        frames: vec![frame],
                    },
                );
                if is_new {
                    self.tiles_order.push(id);
                }
            }
            Atom::BuildAtlas => {
                self.build_atlas();
            }
            Atom::AddPoly {
                id,
                tile_id,
                vertices,
                uvs,
                indices,
                material_id,
            } => {
                let chunk_id = match self.current_chunk {
                    Some(cid) => cid,
                    None => {
                        let cid = Uuid::new_v4();
                        self.chunks_map.insert(cid, Chunk::default());
                        self.current_chunk = Some(cid);
                        cid
                    }
                };
                self.chunks_map.entry(chunk_id).or_default().add_poly_2d(
                    id,
                    tile_id,
                    vertices,
                    uvs,
                    indices,
                    self.current_layer,
                    true,
                    material_id,
                );
            }
            Atom::AddPoly3D {
                id,
                tile_id,
                vertices,
                uvs,
                indices,
                material_id,
            } => {
                let chunk_id = match self.current_chunk {
                    Some(cid) => cid,
                    None => {
                        let cid = Uuid::new_v4();
                        self.chunks_map.insert(cid, Chunk::default());
                        self.current_chunk = Some(cid);
                        cid
                    }
                };
                self.chunks_map.entry(chunk_id).or_default().add_poly_3d(
                    id,
                    tile_id,
                    vertices,
                    uvs,
                    indices,
                    self.current_layer,
                    true,
                    material_id,
                );
                self.accel_dirty = true;
            }
            Atom::AddLineStrip2D {
                id,
                tile_id,
                points,
                width,
                material_id,
            } => {
                if points.len() < 2 {
                    return;
                }
                let chunk_id = match self.current_chunk {
                    Some(cid) => cid,
                    None => {
                        let cid = Uuid::new_v4();
                        self.chunks_map.insert(cid, Chunk::default());
                        self.current_chunk = Some(cid);
                        cid
                    }
                };
                self.chunks_map
                    .entry(chunk_id)
                    .or_default()
                    .add_line_strip_2d(id, tile_id, points, width, self.current_layer, material_id);
                self.accel_dirty = true;
            }
            Atom::NewChunk { id } => {
                self.chunks_map.entry(id).or_insert_with(Chunk::default);
                self.accel_dirty = true;
            }
            Atom::AddChunk { id, chunk } => {
                // Insert or replace the chunk as-is; caller controls current_chunk separately
                self.chunks_map.insert(id, chunk);
                self.accel_dirty = true;
            }
            Atom::RemoveChunk { id } => {
                let was_current = self.current_chunk == Some(id);
                self.chunks_map.remove(&id);
                if was_current {
                    self.current_chunk = None;
                }
                self.accel_dirty = true;
            }
            Atom::RemoveChunkAt { origin } => {
                if let Some((id, _)) = self.chunks_map.iter().find(|(_, ch)| ch.origin == origin) {
                    let id = *id;
                    let was_current = self.current_chunk == Some(id);
                    self.chunks_map.remove(&id);
                    if was_current {
                        self.current_chunk = None;
                    }
                }
                self.accel_dirty = true;
            }
            Atom::SetCurrentChunk { id } => {
                if !self.chunks_map.contains_key(&id) {
                    self.chunks_map.insert(id, Chunk::default());
                }
                self.current_chunk = Some(id);
            }
            Atom::SetAnimationCounter(n) => {
                self.animation_counter = n;
            }
            Atom::SetSource2D(src) => {
                self.source2d = src;
                if let Some(g) = self.gpu.as_mut() {
                    g.compute2d_pipeline = None;
                }
            }
            Atom::SetSource3D(src) => {
                self.source3d = src;
                if let Some(g) = self.gpu.as_mut() {
                    g.compute3d_pipeline = None;
                }
            }
            Atom::SetTransform2D(m) => {
                self.transform2d = m;
            }
            Atom::SetTransform3D(m) => {
                self.transform3d = m;
                self.accel_dirty = true;
            }
            Atom::SetLayer(l) => {
                self.current_layer = l;
            }
            Atom::Clear => {
                self.atlas_map.clear();
                self.tiles_map.clear();
                self.tiles_order.clear();
                self.atlas.data.fill(0);
                self.chunks_map.clear();
                self.current_chunk = None;
                self.animation_counter = 0;
                self.background = Vec4::new(1.0, 0.8, 0.2, 1.0);
                self.gp0 = Vec4::new(0.0, 0.0, 0.0, 0.0);
                self.gp1 = Vec4::new(0.0, 0.0, 0.0, 0.0);
                self.gp2 = Vec4::new(0.0, 0.0, 0.0, 0.0);
                self.render_mode = RenderMode::Compute2D;
            }
            Atom::ClearTiles => {
                // Clear tile-related state and atlas pixels; keep scene/chunks
                self.atlas_map.clear();
                self.tiles_map.clear();
                self.tiles_order.clear();
                self.atlas.data.fill(0);
            }
            Atom::ClearGeometry => {
                // Remove all chunks and unset current chunk; keep tiles/atlas/state
                self.chunks_map.clear();
                self.current_chunk = None;
                self.accel_dirty = true;
            }
            Atom::SetBackground(v) => {
                self.background = v;
            }
            Atom::SetGP0(v) => {
                self.gp0 = v;
            }
            Atom::SetGP1(v) => {
                self.gp1 = v;
            }
            Atom::SetGP2(v) => {
                self.gp2 = v;
            }
            Atom::SetGP3(v) => {
                self.gp3 = v;
            }
            Atom::SetGP4(v) => {
                self.gp4 = v;
            }
            Atom::SetGP5(v) => {
                self.gp5 = v;
            }
            Atom::SetGP6(v) => {
                self.gp6 = v;
            }
            Atom::SetGP7(v) => {
                self.gp7 = v;
            }
            Atom::SetGP8(v) => {
                self.gp8 = v;
            }
            Atom::SetGP9(v) => {
                self.gp9 = v;
            }
            Atom::SetRenderMode(m) => {
                self.render_mode = m;
            }
            Atom::AddLight { id, light } => {
                self.lights.insert(id, light);
            }
            Atom::RemoveLight { id } => {
                self.lights.remove(&id);
            }
            Atom::ClearLights => {
                self.lights.clear();
            }
            Atom::AddMaterial { id, material } => {
                self.materials.insert(id, material);
            }
            Atom::RemoveMaterial { id } => {
                self.materials.remove(&id);
            }
            Atom::ClearMaterials => {
                self.materials.clear();
            }
            Atom::SetGeoMaterial { id, material_id } => {
                for ch in self.chunks_map.values_mut() {
                    if let Some(p) = ch.polys_map.get_mut(&id) {
                        p.material_id = material_id;
                    }
                    if let Some(p3) = ch.polys3d_map.get_mut(&id) {
                        p3.material_id = material_id;
                    }
                }
            }
            Atom::SetSceneGridCells { target_cells } => {
                self.scene_grid_cells = target_cells;
                self.accel_dirty = true;
            }
            Atom::ClearSceneGrid => {
                // Reset to an empty 1x1 grid to keep bindings valid
                self.scene_accel = SceneAccel::default();
                self.accel_dirty = true;
            }
        }
    }

    pub fn init_gpu(&mut self, device: &wgpu::Device) {
        use wgpu::ShaderSource;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scenevm-2d-shader"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SCENEVM_2D_WGSL)),
        });

        let globals_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vm-globals-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Globals>() as _),
                },
                count: None,
            }],
        });

        let atlas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vm-atlas-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vm-2d-pipeline-layout"),
            bind_group_layouts: &[&globals_bgl, &atlas_bgl],
            push_constant_ranges: &[],
        });

        let vbuf_layout = wgpu::VertexBufferLayout {
            array_stride: (4 * std::mem::size_of::<f32>()) as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
        };

        let pipeline_2d = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("vm-2d-pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vbuf_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vm-globals-buffer"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.gpu = Some(VMGpu {
            pipeline_2d,
            globals_buf,
            globals_bgl,
            atlas_bgl,
            globals_bg: None,
            atlas_bg: None,
            vbuf: None,
            ibuf: None,
            index_count: 0,
            sampler,
            compute2d_pipeline: None,
            compute3d_pipeline: None,
            u2d_buf: None,
            u3d_buf: None,
            u2d_bgl: None,
            u3d_bgl: None,
            u2d_bg: None,
            u3d_bg: None,
            v2d_ssbo: None,
            i2d_ssbo: None,
            v3d_ssbo: None,
            i3d_ssbo: None,
            tile_offsets: None,
            tile_counts: None,
            tile_tris: None,
            lights_ssbo: None,
            grid_hdr: None,
            grid_offsets: None,
            grid_counts: None,
            grid_tris: None,
            tri_tile: None,
            tri_layer: None,
            tri_mat3d: None,
            materials_ssbo: None,
            tri_mat2d: None,
        });
    }

    /// Returns a read-only view of the current atlas pixels (RGBA8).
    pub fn atlas_pixels(&self) -> &[u8] {
        &self.atlas.data
    }

    /// Copies the atlas into a destination pixel slice of size (dst_w x dst_h) RGBA8.
    /// Does not resize the destination; only overlaps are copied line-by-line.
    pub fn copy_atlas_to_slice(&self, dst: &mut [u8], dst_w: u32, dst_h: u32) {
        self.atlas.copy_to_slice(dst, dst_w, dst_h);
    }

    /// Upload the CPU atlas to GPU (creates GPU resources if needed).
    pub fn upload_atlas_to_gpu_with(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.atlas.upload_to_gpu_with(device, queue);
    }

    /// Download the atlas from GPU into CPU memory; blocks on native, schedules on wasm.
    pub fn download_atlas_from_gpu_with(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.atlas.download_from_gpu_with(device, queue);
    }

    /// Get the atlas rect for a tile's animation frame. Returns None if the tile wasn't packed.
    pub fn frame_rect(&self, id: &Uuid, anim_frame: u32) -> Option<&AtlasEntry> {
        let rects = self.atlas_map.get(id)?;
        if rects.is_empty() {
            return None;
        }
        let idx = (anim_frame as usize) % rects.len();
        rects.get(idx)
    }

    /// Build the atlas using a simple shelf packer and pack all frames.
    fn build_atlas(&mut self) {
        self.atlas.data.fill(0);
        self.atlas_map.clear();

        let mut pen_x: u32 = 0;
        let mut pen_y: u32 = 0;
        let mut shelf_h: u32 = 0;

        for id in &self.tiles_order {
            // Copy needed metadata in a short scope to avoid holding an immutable borrow
            let (w, h, frames_len) = {
                match self.tiles_map.get(id) {
                    Some(t) => (t.w, t.h, t.frames.len()),
                    None => continue,
                }
            };
            if w == 0 || h == 0 {
                continue;
            }

            let mut rects: Vec<AtlasEntry> = Vec::with_capacity(frames_len);

            for f in 0..frames_len {
                // New shelf if doesn't fit in current row
                if pen_x + w > self.atlas.width {
                    pen_x = 0;
                    pen_y = pen_y.saturating_add(shelf_h);
                    shelf_h = 0;
                }
                // If still doesn't fit vertically, stop packing further frames
                if pen_y + h > self.atlas.height {
                    break;
                }

                shelf_h = shelf_h.max(h);

                // Short-lived borrow just to clone the frame bytes; drop before mutating self
                let frame_owned: Vec<u8> =
                    { self.tiles_map.get(id).expect("tile must exist").frames[f].clone() };
                {
                    let atlas_w = self.atlas.width;
                    let dst = &mut self.atlas.data;
                    VM::blit_rgba_into(dst, atlas_w, &frame_owned, w, h, pen_x, pen_y);
                }

                rects.push(AtlasEntry {
                    x: pen_x,
                    y: pen_y,
                    w,
                    h,
                });
                pen_x = pen_x.saturating_add(w);
            }

            if !rects.is_empty() {
                self.atlas_map.insert(*id, rects);
            }
        }
    }

    /// CPU blit of a tightly-packed RGBA8 tile into the atlas at (dst_x, dst_y)
    /// Writes into `dst` (atlas pixel buffer) directly to avoid borrowing `self` mutably.
    fn blit_rgba_into(
        dst: &mut [u8],
        atlas_w: u32,
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst_x: u32,
        dst_y: u32,
    ) {
        if src.is_empty() {
            return;
        }
        let atlas_w_usize = atlas_w as usize;
        let src_stride = (src_w * 4) as usize;
        let dst_stride = (atlas_w * 4) as usize;
        let row_bytes = (src_w * 4) as usize;
        for row in 0..(src_h as usize) {
            let s_off = row * src_stride;
            let d_row = (dst_y as usize + row) * dst_stride;
            let d_off = d_row + (dst_x as usize) * 4;
            let s_end = s_off + row_bytes;
            let d_end = d_off + row_bytes;
            if s_end <= src.len() && d_end <= dst.len() {
                dst[d_off..d_end].copy_from_slice(&src[s_off..s_end]);
            } else {
                break; // OOB guard
            }
            let _ = atlas_w_usize; // suppress unused warning if optimized differently later
        }
    }

    /// Iterate polygons ready for drawing: always yields all polygons in all chunks (ignores current_chunk).
    pub fn polys_2d(&self) -> impl Iterator<Item = (&Poly2D, Option<&AtlasEntry>)> {
        let anim = self.animation_counter as u32;
        self.chunks_map
            .values()
            .flat_map(|ch| ch.polys_map.values())
            .map(move |p| {
                let rect = self.frame_rect(&p.tile_id, anim);
                (p, rect)
            })
    }
}

impl VM {
    /// Initialize compute pipelines and uniform buffers if not yet present.
    pub fn init_compute(&mut self, device: &wgpu::Device) {
        if self.gpu.is_none() {
            // If render pipeline not initialized yet, do it now to allocate gpu struct
            self.init_gpu(device);
        }
        let g = self.gpu.as_mut().unwrap();

        // Uniform BGLs
        let u2d_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vm-u2d-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // UBO
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // storage image (color)
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // atlas texture (sampled)
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // atlas sampler
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // verts SSBO
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // indices SSBO
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // tile offsets
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // tile counts
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // tile tris
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10, // tri_mat2d
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11, // materials
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let u3d_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vm-u3d-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    // UBO
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // storage image (color)
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // atlas texture (sampled)
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    // atlas sampler
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: verts3d
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 6: indices3d
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 7: grid header (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 8: grid offsets (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 9: grid counts (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 10: grid tris (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 13: tri_mat (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 14: materials SSBO (storage read)
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Pipelines (compile only if missing)
        if g.u2d_bgl.is_none() {
            g.u2d_bgl = Some(u2d_bgl);
        }
        if g.u3d_bgl.is_none() {
            g.u3d_bgl = Some(u3d_bgl);
        }

        if g.compute2d_pipeline.is_none() {
            let src2d = [SCENEVM_2D_HEADER, &self.source2d].concat();
            let cs2d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vm-2d-cs"),
                source: wgpu::ShaderSource::Wgsl(src2d.into()),
            });
            let pl2d = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("vm-2d-cs-pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("vm-2d-cs-layout"),
                        bind_group_layouts: &[g.u2d_bgl.as_ref().unwrap()],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &cs2d,
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            });
            g.compute2d_pipeline = Some(pl2d);
        }

        if g.compute3d_pipeline.is_none() {
            let src3d = [SCENEVM_3D_HEADER, &self.source3d].concat();
            let cs3d = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("vm-3d-cs"),
                source: wgpu::ShaderSource::Wgsl(src3d.into()),
            });
            let pl3d = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("vm-3d-cs-pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("vm-3d-cs-layout"),
                        bind_group_layouts: &[g.u3d_bgl.as_ref().unwrap()],
                        push_constant_ranges: &[],
                    }),
                ),
                module: &cs3d,
                entry_point: Some("cs_main"),
                compilation_options: Default::default(),
                cache: None,
            });
            g.compute3d_pipeline = Some(pl3d);
        }

        // UBOs
        if g.u2d_buf.is_none() {
            let u2d_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vm-u2d"),
                size: std::mem::size_of::<Compute2DUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            g.u2d_buf = Some(u2d_buf);
        }
        if g.u3d_buf.is_none() {
            let u3d_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vm-u3d"),
                size: std::mem::size_of::<Compute3DUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            g.u3d_buf = Some(u3d_buf);
        }
        g.u2d_bg = None;
        g.u3d_bg = None;
    }

    /// Dispatches 2D compute pipeline into a storage-capable surface.
    pub fn compute_draw_2d_into(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &mut Texture,
        fb_w: u32,
        fb_h: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
        self.init_compute(device);
        // Require surface to be STORAGE-capable. If your Texture lacks this, recreate with STORAGE_BINDING.
        surface.ensure_gpu_with(device);
        // Update uniforms
        let m = self.transform2d;
        let m_inv = mat3_inverse_f32(&m).unwrap_or(Mat3::<f32>::identity());
        let u = Compute2DUniforms {
            background: self.background.into_array(),
            fb_size: [fb_w, fb_h],
            _pad0: [0, 0],
            gp0: self.gp0.into_array(),
            gp1: self.gp1.into_array(),
            gp2: self.gp2.into_array(),
            gp3: self.gp3.into_array(),
            gp4: self.gp4.into_array(),
            gp5: self.gp5.into_array(),
            gp6: self.gp6.into_array(),
            gp7: self.gp7.into_array(),
            gp8: self.gp8.into_array(),
            gp9: self.gp9.into_array(),
            // Mat3 columns (col-major), pad .w = 0.0
            mat2d_c0: [m[(0, 0)], m[(1, 0)], m[(2, 0)], 0.0],
            mat2d_c1: [m[(0, 1)], m[(1, 1)], m[(2, 1)], 0.0],
            mat2d_c2: [m[(0, 2)], m[(1, 2)], m[(2, 2)], 0.0],
            mat2d_inv_c0: [m_inv[(0, 0)], m_inv[(1, 0)], m_inv[(2, 0)], 0.0],
            mat2d_inv_c1: [m_inv[(0, 1)], m_inv[(1, 1)], m_inv[(2, 1)], 0.0],
            mat2d_inv_c2: [m_inv[(0, 2)], m_inv[(1, 2)], m_inv[(2, 2)], 0.0],
            lights_count: self.lights.len() as u32,
            _pad_lights_align: [0, 0, 0],
            _pad_lights_vec3: [0, 0, 0, 0],
        };
        if let Some(g) = self.gpu.as_ref() {
            queue.write_buffer(g.u2d_buf.as_ref().unwrap(), 0, bytemuck::bytes_of(&u));
        }
        // Ensure atlas is available for sampling on GPU
        self.atlas.ensure_gpu_with(device);
        self.upload_atlas_to_gpu_with(device, queue);

        // Build transformed 2D geometry (screen-space) and upload to SSBOs
        let mut verts_flat: Vec<Vert2DPod> = Vec::new();
        let mut indices_flat: Vec<u32> = Vec::new();

        // For layer sorting
        #[derive(Clone, Copy)]
        struct TriMeta {
            layer: i32,
            prio: i32,
            ord: u32,
            mat: Option<Uuid>,
        }
        let mut tri_meta: Vec<TriMeta> = Vec::new();
        let mut tri_ord: u32 = 0;

        for (_cid, ch) in &self.chunks_map {
            let prio = ch.priority;
            for poly in ch.polys_map.values() {
                if !poly.visible {
                    continue;
                }
                let rect_opt = self.frame_rect(&poly.tile_id, self.animation_counter as u32);
                let rect = if let Some(r) = rect_opt { r } else { continue };
                let base = verts_flat.len() as u32;
                let atlas_w = self.atlas.width as f32;
                let atlas_h = self.atlas.height as f32;

                for (i, v) in poly.vertices.iter().enumerate() {
                    // Apply local and global transforms
                    let local_p = poly.transform * Vec3::new(v[0], v[1], 1.0);
                    let world_p = self.transform2d * local_p;

                    // Remap UV into atlas space
                    let base_uv = poly.uvs[i];
                    let u = (rect.x as f32 + base_uv[0] * rect.w as f32) / atlas_w;
                    let v = (rect.y as f32 + base_uv[1] * rect.h as f32) / atlas_h;

                    verts_flat.push(Vert2DPod {
                        pos: [world_p.x, world_p.y],
                        uv: [u, v],
                    });
                }

                for &(a, b, c) in &poly.indices {
                    indices_flat.extend_from_slice(&[
                        base + a as u32,
                        base + b as u32,
                        base + c as u32,
                    ]);
                    tri_meta.push(TriMeta {
                        layer: poly.layer,
                        prio,
                        ord: tri_ord,
                        mat: poly.material_id,
                    });
                    tri_ord = tri_ord.wrapping_add(1);
                }
            }
        }

        // --- CPU tiling & binning (8x8 tiles) ---
        let tiles_x = ((fb_w + 7) / 8).max(1);
        let tiles_y = ((fb_h + 7) / 8).max(1);
        let tiles_n = (tiles_x * tiles_y) as usize;

        #[derive(Clone, Copy)]
        struct TriRef {
            tri: u32,
            layer: i32,
            prio: i32,
            ord: u32,
        }
        let mut bins: Vec<Vec<TriRef>> = vec![Vec::new(); tiles_n];

        let tri_count = (indices_flat.len() / 3) as u32;
        for t in 0..tri_count {
            let i0 = indices_flat[(3 * t as usize) + 0] as usize;
            let i1 = indices_flat[(3 * t as usize) + 1] as usize;
            let i2 = indices_flat[(3 * t as usize) + 2] as usize;
            let a = verts_flat[i0].pos;
            let b = verts_flat[i1].pos;
            let c = verts_flat[i2].pos;

            // pixel-space bbox, clamped to framebuffer
            let minx = f32::min(a[0], f32::min(b[0], c[0])).floor().max(0.0) as i32;
            let maxx = f32::max(a[0], f32::max(b[0], c[0])).ceil().min(fb_w as f32) as i32;
            let miny = f32::min(a[1], f32::min(b[1], c[1])).floor().max(0.0) as i32;
            let maxy = f32::max(a[1], f32::max(b[1], c[1])).ceil().min(fb_h as f32) as i32;
            if minx >= maxx || miny >= maxy {
                continue;
            }

            let tx0 = (minx.max(0) as u32) / 8;
            let ty0 = (miny.max(0) as u32) / 8;
            let tx1 = ((maxx.max(0) as u32).saturating_sub(1)) / 8;
            let ty1 = ((maxy.max(0) as u32).saturating_sub(1)) / 8;

            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    let idx = (ty * tiles_x + tx) as usize;
                    let meta = tri_meta[t as usize];
                    bins[idx].push(TriRef {
                        tri: t as u32,
                        layer: meta.layer,
                        prio: meta.prio,
                        ord: meta.ord,
                    });
                }
            }
        }

        // Flatten to offsets/counts/tris
        let mut tile_offsets: Vec<u32> = Vec::with_capacity(tiles_n);
        let mut tile_counts: Vec<u32> = Vec::with_capacity(tiles_n);
        let mut tile_tris: Vec<u32> = Vec::new();
        let mut running: u32 = 0;
        for v in &mut bins {
            tile_offsets.push(running);
            if !v.is_empty() {
                v.sort_by(|a, b| {
                    b.layer
                        .cmp(&a.layer)
                        .then_with(|| b.prio.cmp(&a.prio))
                        // Painter’s algorithm: later-added (higher ord) should be on top.
                        .then_with(|| b.ord.cmp(&a.ord))
                });
                for r in v.iter() {
                    tile_tris.push(r.tri);
                }
            }
            let c = v.len() as u32;
            tile_counts.push(c);
            running += c;
        }

        // Ensure non-zero-sized buffers
        if tile_offsets.is_empty() {
            tile_offsets.push(0);
        }
        if tile_counts.is_empty() {
            tile_counts.push(0);
        }
        if tile_tris.is_empty() {
            tile_tris.push(0);
        }

        use wgpu::util::DeviceExt;
        // ---- Build materials table (slot 0 = default) ----
        let mut mat_index: FxHashMap<Uuid, u32> = FxHashMap::default();
        let mut materials_flat: Vec<MaterialPod> = Vec::new();
        // default material at 0: white tint, 100% opacity, no emission
        mat_index.insert(Uuid::nil(), 0);
        materials_flat.push(MaterialPod {
            tint: [1.0, 1.0, 1.0, 1.0],
            rmoe: [0.5, 0.0, 1.0, 0.0], // roughness=0.5, metallic=0.0, opacity=1.0, emission=0.0
            model: [0.0, 0.0, 0.0, 0.0],
        });
        // assign indices to all currently registered materials
        for (mid, m) in &self.materials {
            let idx = materials_flat.len() as u32;
            mat_index.insert(*mid, idx);
            materials_flat.push(MaterialPod {
                tint: m.tint.into_array(),
                rmoe: [m.roughness, m.metallic, m.opacity, m.emission],
                model: [m.encode_model(), 0.0, 0.0, 0.0],
            });
        }

        // ---- Map each triangle id (absolute in indices_flat/3 order) → material index ----
        // IMPORTANT: tri_mat2d is indexed by *absolute* triangle id coming from tile_tris.
        let tri_count_abs: usize = (indices_flat.len() / 3) as usize;
        let mut tri_mat2d_vec: Vec<u32> = Vec::with_capacity(tri_count_abs);

        for t in 0..tri_count_abs {
            let m_id = tri_meta[t].mat.unwrap_or(Uuid::nil());
            let idx = *mat_index.get(&m_id).unwrap_or(&0);
            tri_mat2d_vec.push(idx);
        }

        // Keep buffer non-empty for wgpu validation (even if there are 0 triangles)
        if tri_mat2d_vec.is_empty() {
            tri_mat2d_vec.push(0);
        }

        // --- Create non-empty GPU buffers ---
        let tri_mat2d_slice: &[u32] = if tri_mat2d_vec.is_empty() {
            &DUMMY_U32_1
        } else {
            &tri_mat2d_vec
        };
        let materials_slice: &[MaterialPod] = if materials_flat.is_empty() {
            // keep at least one material
            &[MaterialPod {
                tint: [1.0, 1.0, 1.0, 1.0],
                rmoe: [0.5, 0.0, 1.0, 0.0],
                model: [0.0, 0.0, 0.0, 0.0],
            }]
        } else {
            &materials_flat
        };

        let tri_mat2d_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-tri-mat2d"),
            contents: bytemuck::cast_slice(tri_mat2d_slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let materials_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-materials-ssbo"),
            contents: bytemuck::cast_slice(materials_slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let g = self.gpu.as_mut().unwrap();
        g.tri_mat2d = Some(tri_mat2d_buf);
        g.materials_ssbo = Some(materials_buf);

        // Ensure non-zero-sized buffers for binding validation
        let vbytes: Vec<u8> = if verts_flat.is_empty() {
            // one dummy Vert2DPod (pos=0, uv=0) -> 16 bytes
            bytemuck::bytes_of(&Vert2DPod {
                pos: [0.0, 0.0],
                uv: [0.0, 0.0],
            })
            .to_vec()
        } else {
            bytemuck::cast_slice(&verts_flat).to_vec()
        };
        let ibytes: Vec<u8> = if indices_flat.is_empty() {
            // one dummy index
            (0u32).to_ne_bytes().to_vec()
        } else {
            bytemuck::cast_slice(&indices_flat).to_vec()
        };
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-2d-verts-ssbo"),
            contents: &vbytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-2d-indices-ssbo"),
            contents: &ibytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let g = self.gpu.as_mut().unwrap();
        g.v2d_ssbo = Some(vbuf);
        g.i2d_ssbo = Some(ibuf);

        // 1) Upload CPU vectors -> GPU buffers
        let tile_offsets_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-2d-tile-offsets"),
            contents: bytemuck::cast_slice(&tile_offsets),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let tile_counts_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-2d-tile-counts"),
            contents: bytemuck::cast_slice(&tile_counts),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let tile_tris_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-2d-tile-tris"),
            contents: bytemuck::cast_slice(&tile_tris),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 2) Keep them on the GPU state
        let g = self.gpu.as_mut().unwrap();
        g.tile_offsets = Some(tile_offsets_buf);
        g.tile_counts = Some(tile_counts_buf);
        g.tile_tris = Some(tile_tris_buf);

        // Lights
        let mut lights_flat: Vec<LightPod> = Vec::with_capacity(self.lights.len().max(1));
        if self.lights.is_empty() {
            lights_flat.push(LightPod {
                header: [0, 0, 0, 0],
                position: [0.0, 0.0, 0.0, 0.0],
                color: [0.0, 0.0, 0.0, 0.0],
                params0: [0.0, 0.0, 0.0, 0.0],
                params1: [0.0, 0.0, 0.0, 0.0],
            });
        } else {
            for (_id, l) in &self.lights {
                let flicker: f32 = if l.flicker > 0.0 {
                    let hash = hash_u32(self.animation_counter as u32);
                    let combined_hash = hash.wrapping_add(
                        (l.position.x as u32 + l.position.y as u32 + l.position.z as u32) * 100,
                    );
                    let flicker_value = (combined_hash as f32 / u32::MAX as f32).clamp(0.0, 1.0);
                    1.0 - flicker_value * l.flicker
                } else {
                    1.0
                };

                lights_flat.push(LightPod {
                    header: [
                        match l.light_type {
                            LightType::Point => 0,
                        },
                        if l.emitting { 1 } else { 0 },
                        0,
                        0,
                    ],
                    position: [l.position.x, l.position.y, l.position.z, 0.0],
                    color: [l.color.x, l.color.y, l.color.z, 0.0],
                    params0: [l.intensity, l.radius, l.start_distance, l.end_distance],
                    params1: [flicker, 0.0, 0.0, 0.0],
                });
            }
        }
        let lights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-lights-ssbo"),
            contents: bytemuck::cast_slice(&lights_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let g = self.gpu.as_mut().unwrap();
        g.lights_ssbo = Some(lights_buf);

        // Build bind group with surface view and atlas, plus 2D geometry SSBOs
        let view = &surface.gpu.as_ref().unwrap().view;
        let atlas_view = &self.atlas.gpu.as_ref().unwrap().view;
        g.u2d_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vm-u2d-bg"),
            layout: g.u2d_bgl.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: g.u2d_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&g.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: g.v2d_ssbo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: g.i2d_ssbo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: g.tile_offsets.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: g.tile_counts.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: g.tile_tris.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: g.lights_ssbo.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: g.tri_mat2d.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: g.materials_ssbo.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }));
        // Dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vm-2d-cs-enc"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vm-2d-cs-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(g.compute2d_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, g.u2d_bg.as_ref().unwrap(), &[]);
            let gx = (fb_w + 7) / 8;
            let gy = (fb_h + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    /// Dispatches 3D compute pipeline into a storage-capable surface.
    pub fn compute_draw_3d_into(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &mut Texture,
        fb_w: u32,
        fb_h: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
        self.init_compute(device);
        surface.ensure_gpu_with(device);

        // --- Uniforms ---
        let m = self.transform3d;
        let u = Compute3DUniforms {
            background: self.background.into_array(),
            fb_size: [fb_w, fb_h],
            _pad0: [0, 0],
            gp0: self.gp0.into_array(),
            gp1: self.gp1.into_array(),
            gp2: self.gp2.into_array(),
            gp3: self.gp3.into_array(),
            gp4: self.gp4.into_array(),
            gp5: self.gp5.into_array(),
            gp6: self.gp6.into_array(),
            gp7: self.gp7.into_array(),
            gp8: self.gp8.into_array(),
            gp9: self.gp9.into_array(),
            mat3d_c0: [m[(0, 0)], m[(1, 0)], m[(2, 0)], m[(3, 0)]],
            mat3d_c1: [m[(0, 1)], m[(1, 1)], m[(2, 1)], m[(3, 1)]],
            mat3d_c2: [m[(0, 2)], m[(1, 2)], m[(2, 2)], m[(3, 2)]],
            mat3d_c3: [m[(0, 3)], m[(1, 3)], m[(2, 3)], m[(3, 3)]],
            lights_count: self.lights.len() as u32,
            _pad_lights_align: [0, 0, 0],
            _pad_lights_vec3: [0, 0, 0, 0],
        };
        if let Some(g) = self.gpu.as_ref() {
            queue.write_buffer(g.u3d_buf.as_ref().unwrap(), 0, bytemuck::bytes_of(&u));
        }

        // --- Lights ---
        let mut lights_flat: Vec<LightPod> = Vec::with_capacity(self.lights.len().max(1));
        if self.lights.is_empty() {
            lights_flat.push(LightPod {
                header: [0, 0, 0, 0],
                position: [0.0, 0.0, 0.0, 0.0],
                color: [0.0, 0.0, 0.0, 0.0],
                params0: [0.0, 0.0, 0.0, 0.0],
                params1: [0.0, 0.0, 0.0, 0.0],
            });
        } else {
            for (_id, l) in &self.lights {
                let flicker: f32 = if l.flicker > 0.0 {
                    let hash = hash_u32(self.animation_counter as u32);
                    let combined_hash = hash.wrapping_add(
                        (l.position.x as u32 + l.position.y as u32 + l.position.z as u32) * 100,
                    );
                    let flicker_value = (combined_hash as f32 / u32::MAX as f32).clamp(0.0, 1.0);
                    1.0 - flicker_value * l.flicker
                } else {
                    1.0
                };

                lights_flat.push(LightPod {
                    header: [
                        match l.light_type {
                            LightType::Point => 0,
                        },
                        if l.emitting { 1 } else { 0 },
                        0,
                        0,
                    ],
                    position: [l.position.x, l.position.y, l.position.z, 0.0],
                    color: [l.color.x, l.color.y, l.color.z, 0.0],
                    params0: [l.intensity, l.radius, l.start_distance, l.end_distance],
                    params1: [flicker, 0.0, 0.0, 0.0],
                });
            }
        }
        let lights_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-lights-ssbo"),
            contents: bytemuck::cast_slice(&lights_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        {
            let g = self.gpu.as_mut().unwrap();
            g.lights_ssbo = Some(lights_buf);
        }

        // Ensure atlas is available for sampling on GPU
        self.atlas.ensure_gpu_with(device);
        self.upload_atlas_to_gpu_with(device, queue);

        // --- Build 3D geometry (world space) and upload to SSBOs ---
        let mut v3: Vec<Vert3DPod> = Vec::new();
        let mut i3: Vec<u32> = Vec::new();
        let mut tri_mat: Vec<u32> = Vec::new();

        use std::collections::HashMap;
        let mut mat_index: HashMap<Uuid, u32> = HashMap::new();
        let mut materials_vec: Vec<MaterialPod> = Vec::new();

        // slot 0: default material
        materials_vec.push(MaterialPod {
            tint: Vec4::one().into_array(),
            rmoe: [0.5, 0.0, 1.0, 0.0],
            model: [0.0, 0.0, 0.0, 0.0],
        });

        let mut ensure_mat_index = |id_opt: Option<Uuid>| -> u32 {
            match id_opt {
                Some(id) => {
                    if let Some(&idx) = mat_index.get(&id) {
                        idx
                    } else if let Some(m) = self.materials.get(&id) {
                        let idx = materials_vec.len() as u32;
                        mat_index.insert(id, idx);
                        materials_vec.push(MaterialPod {
                            tint: m.tint.into_array(),
                            rmoe: [m.roughness, m.metallic, m.opacity, m.emission],
                            model: [m.encode_model(), 0.0, 0.0, 0.0],
                        });
                        idx
                    } else {
                        0
                    }
                }
                None => 0,
            }
        };

        for (_cid, ch) in &self.chunks_map {
            for poly in ch.polys3d_map.values() {
                if !poly.visible {
                    continue;
                }

                let rect = match self.frame_rect(&poly.tile_id, self.animation_counter as u32) {
                    Some(r) => r,
                    None => continue,
                };

                let vcount = poly.vertices.len();
                let mut poly_pos: Vec<[f32; 3]> = Vec::with_capacity(vcount);
                let mut poly_nrm: Vec<[f32; 3]> = vec![[0.0, 0.0, 0.0]; vcount];

                for v in &poly.vertices {
                    let p = m * Vec4::new(v[0], v[1], v[2], v[3]);
                    let w = if p.w != 0.0 { p.w } else { 1.0 };
                    poly_pos.push([p.x / w, p.y / w, p.z / w]);
                }

                for &(a, b, c) in &poly.indices {
                    let pa = poly_pos[a];
                    let pb = poly_pos[b];
                    let pc = poly_pos[c];
                    let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                    let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                    let nx = e1[1] * e2[2] - e1[2] * e2[1];
                    let ny = e1[2] * e2[0] - e1[0] * e2[2];
                    let nz = e1[0] * e2[1] - e1[1] * e2[0];
                    poly_nrm[a][0] += nx;
                    poly_nrm[a][1] += ny;
                    poly_nrm[a][2] += nz;
                    poly_nrm[b][0] += nx;
                    poly_nrm[b][1] += ny;
                    poly_nrm[b][2] += nz;
                    poly_nrm[c][0] += nx;
                    poly_nrm[c][1] += ny;
                    poly_nrm[c][2] += nz;
                }
                for n in &mut poly_nrm {
                    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                    if len > 1e-12 {
                        n[0] /= len;
                        n[1] /= len;
                        n[2] /= len;
                    }
                }

                let base = v3.len() as u32;
                let atlas_w = self.atlas.width as f32;
                let atlas_h = self.atlas.height as f32;

                for (i, p) in poly_pos.iter().enumerate() {
                    let uv0 = poly.uvs[i];
                    let u = (rect.x as f32 + uv0[0] * rect.w as f32) / atlas_w;
                    let v_uv = (rect.y as f32 + uv0[1] * rect.h as f32) / atlas_h;
                    let n = poly_nrm[i];
                    v3.push(Vert3DPod {
                        pos: [p[0], p[1], p[2]],
                        _pad_pos: 0.0,
                        uv: [u, v_uv],
                        _pad_uv: [0.0, 0.0],
                        normal: [n[0], n[1], n[2]],
                        _pad_n: 0.0,
                    });
                }
                let mat_slot = ensure_mat_index(poly.material_id);
                for &(a, b, c) in &poly.indices {
                    i3.extend_from_slice(&[base + a as u32, base + b as u32, base + c as u32]);
                    tri_mat.push(mat_slot);
                }
            }
        }

        if v3.is_empty() {
            v3.push(Vert3DPod {
                pos: [0.0; 3],
                _pad_pos: 0.0,
                uv: [0.0; 2],
                _pad_uv: [0.0; 2],
                normal: [0.0, 0.0, 1.0],
                _pad_n: 0.0,
            });
        }
        if i3.is_empty() {
            i3.push(0);
        }

        if self.accel_dirty {
            // Build a grid from the actual uploaded geometry so slices won't be empty.
            self.build_scene_grid_from(&v3, &i3, 0.0, self.scene_grid_cells); //200_000);
            self.accel_dirty = false;
        }

        // --- Upload scene-wide grid to GPU (always present) ---
        use wgpu::util::DeviceExt;
        let gr = &self.scene_accel.grid;

        // Make sure header dims are never 0 to avoid div-by-zero in WGSL.
        let hdr = Grid3DHeader {
            origin: [gr.origin.x, gr.origin.y, gr.origin.z, 0.0],
            cell_size: [gr.cell_size.x, gr.cell_size.y, gr.cell_size.z, 0.0],
            dims: [gr.dims[0].max(1), gr.dims[1].max(1), gr.dims[2].max(1), 0],
        };

        // wgpu forbids binding zero-sized STORAGE buffers. Use non-empty dummy slices if empty.
        let cell_offsets_slice: &[u32] = if gr.cell_offsets.is_empty() {
            &DUMMY_U32_1
        } else {
            &gr.cell_offsets
        };
        let cell_counts_slice: &[u32] = if gr.cell_counts.is_empty() {
            &DUMMY_U32_1
        } else {
            &gr.cell_counts
        };
        let cell_tris_slice: &[u32] = if gr.cell_tris.is_empty() {
            &DUMMY_U32_1
        } else {
            &gr.cell_tris
        };
        let tri_tile_slice: &[u32] = if gr.tri_tile.is_empty() {
            &DUMMY_U32_1
        } else {
            &gr.tri_tile
        };
        let tri_layer_slice: &[i32] = if gr.tri_layer.is_empty() {
            &DUMMY_I32_1
        } else {
            &gr.tri_layer
        };

        let grid_hdr = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-grid3d-hdr"),
            contents: bytemuck::bytes_of(&hdr),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let grid_offsets = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-grid3d-offsets"),
            contents: bytemuck::cast_slice(cell_offsets_slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let grid_counts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-grid3d-counts"),
            contents: bytemuck::cast_slice(cell_counts_slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let grid_tris = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-grid3d-tris"),
            contents: bytemuck::cast_slice(cell_tris_slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let tri_tile = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-tri-tile"),
            contents: bytemuck::cast_slice(tri_tile_slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let tri_layer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-tri-layer"),
            contents: bytemuck::cast_slice(tri_layer_slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        {
            // short mutable borrow of self.gpu
            let g = self.gpu.as_mut().unwrap();
            g.grid_hdr = Some(grid_hdr);
            g.grid_offsets = Some(grid_offsets);
            g.grid_counts = Some(grid_counts);
            g.grid_tris = Some(grid_tris);
            g.tri_tile = Some(tri_tile);
            g.tri_layer = Some(tri_layer);
        }

        let v3_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-3d-verts-ssbo"),
            contents: bytemuck::cast_slice(&v3),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let i3_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-3d-indices-ssbo"),
            contents: bytemuck::cast_slice(&i3),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let tri_mat_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-3d-tri-mat"),
            contents: bytemuck::cast_slice(if tri_mat.is_empty() {
                &[0u32][..]
            } else {
                &tri_mat[..]
            }),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let materials_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vm-materials-ssbo"),
            contents: bytemuck::cast_slice(if materials_vec.is_empty() {
                &[MaterialPod {
                    tint: [1.0, 1.0, 1.0, 1.0],
                    rmoe: [0.5, 0.0, 1.0, 0.0],
                    model: [0.0, 0.0, 0.0, 0.0],
                }][..]
            } else {
                &materials_vec[..]
            }),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let g = self.gpu.as_mut().unwrap();
        g.tri_mat3d = Some(tri_mat_buf);
        g.materials_ssbo = Some(materials_buf);
        {
            let g = self.gpu.as_mut().unwrap();
            g.v3d_ssbo = Some(v3_buf);
            g.i3d_ssbo = Some(i3_buf);
        }

        // Avoid borrowing self immutably while we need &mut for bind group creation.
        let surface_view = surface.gpu.as_ref().unwrap().view.clone();
        let atlas_view = self.atlas.gpu.as_ref().unwrap().view.clone();

        // Build the bind group
        {
            let g = self.gpu.as_mut().unwrap();
            g.u3d_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vm-u3d-bg"),
                layout: g.u3d_bgl.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: g.u3d_buf.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&surface_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&atlas_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&g.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: g.lights_ssbo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: g.v3d_ssbo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: g.i3d_ssbo.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: g.grid_hdr.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: g.grid_offsets.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: g.grid_counts.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: g.grid_tris.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: g.tri_mat3d.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: g.materials_ssbo.as_ref().unwrap().as_entire_binding(),
                    },
                ],
            }));
        }

        // Dispatch
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vm-3d-cs-enc"),
        });
        {
            let g = self.gpu.as_ref().unwrap();
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vm-3d-cs-pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(g.compute3d_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, g.u3d_bg.as_ref().unwrap(), &[]);
            let gx = (fb_w + 7) / 8;
            let gy = (fb_h + 7) / 8;
            cpass.dispatch_workgroups(gx, gy, 1);
        }
        queue.submit(Some(encoder.finish()));
    }

    fn build_scene_grid_from(
        &mut self,
        verts: &[Vert3DPod],
        indices: &[u32],
        cell_world: f32,
        target_cells: u32,
    ) {
        use vek::Vec3;

        #[inline(always)]
        fn vmin(a: vek::Vec3<f32>, b: vek::Vec3<f32>) -> vek::Vec3<f32> {
            vek::Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
        }
        #[inline(always)]
        fn vmax(a: vek::Vec3<f32>, b: vek::Vec3<f32>) -> vek::Vec3<f32> {
            vek::Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
        }

        // --- 1) Scene AABB (over all positions) ---
        let mut bmin = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut bmax = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for v in verts {
            let p = Vec3::new(v.pos[0], v.pos[1], v.pos[2]);
            bmin = vmin(bmin, p);
            bmax = vmax(bmax, p);
        }

        // Empty scene guard
        if !bmin.x.is_finite() {
            // Make a 1x1x1 dummy grid so bindings are valid
            self.scene_accel.grid = SceneGridAccel {
                origin: Vec3::zero(),
                cell_size: Vec3::broadcast(1.0),
                dims: [1, 1, 1],
                cell_offsets: vec![0],
                cell_counts: vec![0],
                cell_tris: vec![0],
                tri_tile: vec![0],
                tri_layer: vec![0],
            };
            return;
        }

        // --- 2) Pad scene AABB slightly ---
        let diag = (bmax - bmin).magnitude().max(1e-6);
        let pad = 0.00001 * diag; // scene padding
        bmin -= Vec3::broadcast(pad);
        bmax += Vec3::broadcast(pad);

        // --- 3) Choose grid resolution ---
        let size = bmax - bmin;
        let vol = (size.x * size.y * size.z).max(1e-9);
        let nx: u32;
        let ny: u32;
        let nz;

        if cell_world > 0.0 {
            // User-enforced cell size
            nx = (size.x / cell_world).ceil().max(1.0) as u32;
            ny = (size.y / cell_world).ceil().max(1.0) as u32;
            nz = (size.z / cell_world).ceil().max(1.0) as u32;
        } else {
            // Aim for target_cells
            let t = target_cells.max(1) as f32;
            let s = (vol / t).cbrt(); // cubic cell size
            nx = (size.x / s).ceil().max(1.0) as u32;
            ny = (size.y / s).ceil().max(1.0) as u32;
            nz = (size.z / s).ceil().max(1.0) as u32;
        }

        let dims = [nx, ny, nz];
        let cell_size = Vec3::new(
            size.x / nx.max(1) as f32,
            size.y / ny.max(1) as f32,
            size.z / nz.max(1) as f32,
        );

        // Storage for CSR
        let cell_count = (nx as usize) * (ny as usize) * (nz as usize);
        let mut cell_vecs: Vec<Vec<u32>> = vec![Vec::new(); cell_count];

        // Precompute an epsilon in **world** based on cell size (robustness)
        let cell_eps = cell_size.x.max(cell_size.y).max(cell_size.z) * 0.8;

        // --- 4) Bin triangles into cells with **padded tri AABB** ---
        let tri_count = indices.len() / 3;
        for tri in 0..tri_count {
            let i0 = indices[3 * tri + 0] as usize;
            let i1 = indices[3 * tri + 1] as usize;
            let i2 = indices[3 * tri + 2] as usize;

            let p0 = Vec3::new(verts[i0].pos[0], verts[i0].pos[1], verts[i0].pos[2]);
            let p1 = Vec3::new(verts[i1].pos[0], verts[i1].pos[1], verts[i1].pos[2]);
            let p2 = Vec3::new(verts[i2].pos[0], verts[i2].pos[1], verts[i2].pos[2]);

            // Triangle AABB (then pad!)
            let mut tmin = vmin(vmin(p0, p1), p2);
            let mut tmax = vmax(vmax(p0, p1), p2);
            tmin -= Vec3::broadcast(cell_eps);
            tmax += Vec3::broadcast(cell_eps);

            // Map to cell coords
            let rel_min = (tmin - bmin) / cell_size;
            let rel_max = (tmax - bmin) / cell_size;

            let mut ix0 = rel_min.x.floor() as i32;
            let mut iy0 = rel_min.y.floor() as i32;
            let mut iz0 = rel_min.z.floor() as i32;
            let mut ix1 = rel_max.x.ceil() as i32;
            let mut iy1 = rel_max.y.ceil() as i32;
            let mut iz1 = rel_max.z.ceil() as i32;

            // Clamp to grid
            ix0 = ix0.clamp(0, nx as i32 - 1);
            iy0 = iy0.clamp(0, ny as i32 - 1);
            iz0 = iz0.clamp(0, nz as i32 - 1);
            ix1 = ix1.clamp(0, nx as i32 - 1);
            iy1 = iy1.clamp(0, ny as i32 - 1);
            iz1 = iz1.clamp(0, nz as i32 - 1);

            if ix0 > ix1 || iy0 > iy1 || iz0 > iz1 {
                continue;
            }

            // Fill all overlapped cells
            for z in iz0..=iz1 {
                for y in iy0..=iy1 {
                    for x in ix0..=ix1 {
                        let idx = ((z as u32 * ny + y as u32) * nx + x as u32) as usize;
                        cell_vecs[idx].push(tri as u32);
                    }
                }
            }
        }

        // --- 5) CSR flatten (stable order) ---
        let mut offsets = vec![0u32; cell_count];
        let mut counts = vec![0u32; cell_count];
        let mut tris: Vec<u32> = Vec::new();
        tris.reserve(cell_vecs.iter().map(|v| v.len()).sum());

        let mut run = 0u32;
        for (i, v) in cell_vecs.iter_mut().enumerate() {
            offsets[i] = run;
            // sort for determinism (optional)
            v.sort_unstable();
            v.dedup(); // optional de-dup if your binning can push the same tri twice
            run += v.len() as u32;
            counts[i] = v.len() as u32;
            tris.extend(v.iter().copied());
        }

        // --- 6) Per-triangle metadata (optional; keep aligned length) ---
        let tri_tile = vec![0u32; tri_count.max(1)];
        let tri_layer = vec![0i32; tri_count.max(1)];

        // --- 7) Store on the VM ---
        self.scene_accel.grid = SceneGridAccel {
            origin: bmin,
            cell_size,
            dims,
            cell_offsets: if offsets.is_empty() { vec![0] } else { offsets },
            cell_counts: if counts.is_empty() { vec![0] } else { counts },
            cell_tris: if tris.is_empty() { vec![0] } else { tris },
            tri_tile,
            tri_layer,
        };
    }

    /// Unified draw entry: chooses 2D or 3D compute path based on `self.render_mode`.
    pub fn draw_into(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &mut Texture,
        fb_w: u32,
        fb_h: u32,
    ) {
        match self.render_mode {
            RenderMode::Compute2D => self.compute_draw_2d_into(device, queue, surface, fb_w, fb_h),
            RenderMode::Compute3D => self.compute_draw_3d_into(device, queue, surface, fb_w, fb_h),
        }
    }
} // end impl VM

// Helper for inverting a 3x3 matrix (vek::Mat3<f32>)
fn mat3_inverse_f32(m: &Mat3<f32>) -> Option<Mat3<f32>> {
    // Treat elements as a standard 3x3 laid out by rows using vek indexing (col, row)
    let a = m[(0, 0)];
    let b = m[(1, 0)];
    let c = m[(2, 0)];
    let d = m[(0, 1)];
    let e = m[(1, 1)];
    let f = m[(2, 1)];
    let g = m[(0, 2)];
    let h = m[(1, 2)];
    let i = m[(2, 2)];

    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() < 1e-8 {
        return None;
    }
    let inv_det = 1.0 / det;

    let m00 = (e * i - f * h) * inv_det;
    let m01 = (c * h - b * i) * inv_det;
    let m02 = (b * f - c * e) * inv_det;

    let m10 = (f * g - d * i) * inv_det;
    let m11 = (a * i - c * g) * inv_det;
    let m12 = (c * d - a * f) * inv_det;

    let m20 = (d * h - e * g) * inv_det;
    let m21 = (b * g - a * h) * inv_det;
    let m22 = (a * e - b * d) * inv_det;

    let mut out = Mat3::<f32>::zero();
    // Write back using vek's (col,row) indexing
    out[(0, 0)] = m00;
    out[(1, 0)] = m01;
    out[(2, 0)] = m02;
    out[(0, 1)] = m10;
    out[(1, 1)] = m11;
    out[(2, 1)] = m12;
    out[(0, 2)] = m20;
    out[(1, 2)] = m21;
    out[(2, 2)] = m22;
    Some(out)
}

/// Hash for light flickering
fn hash_u32(mut state: u32) -> u32 {
    state = (state ^ 61) ^ (state >> 16);
    state = state.wrapping_add(state << 3);
    state ^= state >> 4;
    state = state.wrapping_mul(0x27d4eb2d);
    state ^= state >> 15;
    state
}
