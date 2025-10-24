use crate::Texture;
use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use uuid::Uuid;
use vek::Vec4;
use vek::{Mat3, Vec3};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]

/// The Geometry Identifier for polygons and triangles.
pub enum GeoId {
    Unknown(u32),
    Vertex(u32),
    Linedef(u32),
    Sector(u32),
    Character(u32),
    Item(u32),
    Triangle(u32),
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vert2DPod {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
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
    },
    /// Add a simple 2D line strip as thick segments tessellated into quads (no caps/joins)
    /// Points are in world coordinates; width is in the same units.
    AddLineStrip2D {
        id: GeoId,
        tile_id: Uuid,
        points: Vec<[f32; 2]>,
        width: f32,
    },
    /// Create an empty chunk (no switch)
    NewChunk {
        id: Uuid,
    },
    /// Insert or replace an entire chunk in one go (prepared externally, e.g., in Rusterix)
    SetChunk {
        id: Uuid,
        chunk: Chunk,
    },
    /// Remove an existing chunk; if it is the current chunk, unset it
    RemoveChunk {
        id: Uuid,
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
    /// Switch between 2D and 3D compute drawing
    SetRenderMode(RenderMode),
    /// Set a 2D transform (Mat3) applied on CPU to polygon vertices before 2D compute draw
    SetTransform2D(Mat3<f32>),
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
    pub indices: Vec<(usize, usize, usize)>, // triangle list, LOCAL to its chunk (Rusterix-compatible)
    pub transform: Mat3<f32>,                // per-poly local transform
}

#[derive(Debug, Default, Clone)]
pub struct Chunk {
    pub polys_map: FxHashMap<GeoId, Poly2D>,
    pub priority: i32,
}

impl Chunk {
    /// Add a 2D polygon with explicit vertices/uvs/indices. Indices are local to this chunk.
    pub fn add_poly_2d(
        &mut self,
        id: GeoId,
        tile_id: Uuid,
        vertices: Vec<[f32; 2]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<(usize, usize, usize)>,
    ) {
        let poly = Poly2D {
            id,
            tile_id,
            vertices,
            uvs,
            indices,
            transform: Mat3::identity(),
        };
        self.polys_map.insert(id, poly);
    }

    /// Add a 2D line strip tessellated into thick quads (no caps/joins) as one poly.
    /// `points` are in world coords; `width` is in world units.
    pub fn add_line_strip_2d(
        &mut self,
        id: GeoId,
        tile_id: Uuid,
        points: Vec<[f32; 2]>,
        width: f32,
    ) {
        if points.len() < 2 {
            return;
        }
        let half = 0.5 * width;
        let mut vertices: Vec<[f32; 2]> = Vec::with_capacity(points.len() * 4);
        let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(points.len() * 4);
        let mut indices: Vec<(usize, usize, usize)> = Vec::with_capacity((points.len() - 1) * 2);

        for seg in 0..(points.len() - 1) {
            let p0 = points[seg];
            let p1 = points[seg + 1];
            let dx = p1[0] - p0[0];
            let dy = p1[1] - p0[1];
            let len = (dx * dx + dy * dy).sqrt();
            if len == 0.0 {
                continue;
            }
            let nx = -dy / len; // left-hand normal (perp)
            let ny = dx / len;
            let ox = nx * half;
            let oy = ny * half;

            // Quad corners (consistent winding: 0-1-2, 0-2-3)
            let v0 = [p0[0] - ox, p0[1] - oy]; // bottom-left
            let v1 = [p0[0] + ox, p0[1] + oy]; // top-left
            let v2 = [p1[0] + ox, p1[1] + oy]; // top-right
            let v3 = [p1[0] - ox, p1[1] - oy]; // bottom-right

            let base = vertices.len();
            vertices.extend_from_slice(&[v0, v1, v2, v3]);
            // Simple UVs per quad (stretch along segment)
            uvs.extend_from_slice(&[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]);
            indices.push((base + 0, base + 1, base + 2));
            indices.push((base + 0, base + 2, base + 3));
        }

        if vertices.is_empty() {
            return;
        }

        let poly = Poly2D {
            id,
            tile_id,
            vertices,
            uvs,
            indices,
            transform: Mat3::identity(),
        };
        self.polys_map.insert(id, poly);
    }

    /// Add a square (axis-aligned) centered at `center` with edge length `size`.
    /// Inserts a new Poly2D using `tile_id` and `id`. UVs cover the full tile.
    pub fn add_square_2d(&mut self, id: GeoId, tile_id: Uuid, center: [f32; 2], size: f32) {
        if size <= 0.0 {
            return;
        }
        let half = 0.5 * size;
        let (cx, cy) = (center[0], center[1]);
        let x0 = cx - half; // left
        let x1 = cx + half; // right
        let y0 = cy - half; // bottom
        let y1 = cy + half; // top

        let vertices = vec![
            [x0, y0], // bottom-left
            [x0, y1], // top-left
            [x1, y1], // top-right
            [x1, y0], // bottom-right
        ];
        let uvs = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        let indices = vec![(0, 1, 2), (0, 2, 3)];

        let poly = Poly2D {
            id,
            tile_id,
            vertices,
            uvs,
            indices,
            transform: Mat3::identity(),
        };
        self.polys_map.insert(id, poly);
    }
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
    // --- Tiling
    pub tile_offsets: Option<wgpu::Buffer>,
    pub tile_counts: Option<wgpu::Buffer>,
    pub tile_tris: Option<wgpu::Buffer>,
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
    // Mat3<f32> as 3 padded vec4 columns (col-major), .w is padding
    pub mat2d_c0: [f32; 4],
    pub mat2d_c1: [f32; 4],
    pub mat2d_c2: [f32; 4],
    // Inverse 2D matrix columns
    pub mat2d_inv_c0: [f32; 4],
    pub mat2d_inv_c1: [f32; 4],
    pub mat2d_inv_c2: [f32; 4],
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
    // Mat4<f32> as 4 vec4 columns (col-major)
    pub mat3d_c0: [f32; 4],
    pub mat3d_c1: [f32; 4],
    pub mat3d_c2: [f32; 4],
    pub mat3d_c3: [f32; 4],
}

pub const SCENEVM_2D_CS_WGSL: &str = r#"
struct U2D { background: vec4<f32>, fb_size: vec2<u32>, _pad: vec2<u32>, };
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
  mat2d_c0: vec4<f32>,
  mat2d_c1: vec4<f32>,
  mat2d_c2: vec4<f32>,
  mat2d_inv_c0: vec4<f32>,
  mat2d_inv_c1: vec4<f32>,
  mat2d_inv_c2: vec4<f32>,
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

fn tiles_x() -> u32 { return (U.fb_size.x + 7u) / 8u; }
fn tiles_y() -> u32 { return (U.fb_size.y + 7u) / 8u; }
fn tile_index(tx: u32, ty: u32) -> u32 { return ty * tiles_x() + tx; }

fn sv_write(px: u32, py: u32, c: vec4<f32>) {
  textureStore(color_out, vec2<i32>(i32(px), i32(py)), c);
}
fn sv_sample(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(atlas_tex, atlas_smp, uv, 0.0);
}
// ----- SceneVM 2D helpers -----
struct BaryHit { hit: bool, w: vec3<f32> };
struct ColorHit { hit: bool, color: vec4<f32> };

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

fn sv_tri_color(p: vec2<f32>, i0: u32, i1: u32, i2: u32) -> ColorHit {
  let a = verts.data[i0].pos;
  let b = verts.data[i1].pos;
  let c = verts.data[i2].pos;
  let bh = sv_tri_bary(p, a, b, c);
  if (!bh.hit) { return ColorHit(false, vec4<f32>(0.0)); }
  let w = bh.w;
  let uv = verts.data[i0].uv * w.x + verts.data[i1].uv * w.y + verts.data[i2].uv * w.z;
  let col = sv_sample(uv);
  return ColorHit(true, col);
}

fn sv_world_from_screen(pix: vec2<f32>) -> vec2<f32> {
  let invM = mat3x3<f32>(U.mat2d_inv_c0.xyz, U.mat2d_inv_c1.xyz, U.mat2d_inv_c2.xyz);
  let v = invM * vec3<f32>(pix, 1.0);
  return v.xy;
}
fn sv_shade_tile_pixel(p: vec2<f32>, px: u32, py: u32, tid: u32) -> bool {
  let off = tile_offsets.data[tid];
  let cnt = tile_counts.data[tid];
  for (var k: u32 = 0u; k < cnt; k = k + 1u) {
    let t  = tile_tris.data[off + k];
    let i0 = indices.data[3u*t + 0u];
    let i1 = indices.data[3u*t + 1u];
    let i2 = indices.data[3u*t + 2u];
    let ch = sv_tri_color(p, i0, i1, i2);
    if (ch.hit) {
      sv_write(px, py, ch.color);
      return true;
    }
  }
  return false;
}
// ----- end helpers -----
"#;

pub const SCENEVM_3D_HEADER: &str = r#"
struct U3D {
  background: vec4<f32>,
  fb_size: vec2<u32>, _pad0: vec2<u32>,
  gp0: vec4<f32>, gp1: vec4<f32>, gp2: vec4<f32>,
  mat3d_c0: vec4<f32>,
  mat3d_c1: vec4<f32>,
  mat3d_c2: vec4<f32>,
  mat3d_c3: vec4<f32>,
};
@group(0) @binding(0) var<uniform> U: U3D;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var atlas_tex: texture_2d<f32>;
@group(0) @binding(3) var atlas_smp: sampler;

fn sv_write(px: u32, py: u32, c: vec4<f32>) {
  textureStore(color_out, vec2<i32>(i32(px), i32(py)), c);
}
fn sv_sample(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(atlas_tex, atlas_smp, uv, 0.0);
}
"#;

pub const DEFAULT_2D_BODY: &str = r#"
@compute @workgroup_size(8,8,1)
fn cs_main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let px = wg.x * 8u + lid.x;
  let py = wg.y * 8u + lid.y;
  if (px >= U.fb_size.x || py >= U.fb_size.y) { return; }

  //let uv = vec2<f32>(f32(gid.x)/f32(U.fb_size.x), f32(gid.y)/f32(U.fb_size.y));

  // Convert pixel coordinates into world coordinates
  //let world_pos = sv_world_from_screen(vec2<f32>(f32(px), f32(py)));  

  let p = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

  // Clear to background first
  sv_write(px, py, U.background);

  let tid = tile_index(wg.x, wg.y);
  if (sv_shade_tile_pixel(p, px, py, tid)) { return; }
}
"#;

pub const DEFAULT_3D_BODY: &str = r#"
@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= U.fb_size.x || gid.y >= U.fb_size.y) { return; }
  let uv = vec2<f32>(f32(gid.x)/f32(U.fb_size.x), f32(gid.y)/f32(U.fb_size.y));
  let b = U.background.x;
  let col = vec4<f32>(uv.x*b, uv.y*b, b, 1.0);
  sv_write(gid.x, gid.y, col);
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
    // --- Programmable compute shader sources
    pub source2d: String,
    pub source3d: String,
    pub transform2d: Mat3<f32>,
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
            source2d: DEFAULT_2D_BODY.to_string(),
            source3d: DEFAULT_3D_BODY.to_string(),
            transform2d: Mat3::identity(),
        }
    }

    /// Interpret one instruction.
    pub fn execute(&mut self, atom: Atom) {
        match atom {
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
                self.chunks_map
                    .entry(chunk_id)
                    .or_default()
                    .add_poly_2d(id, tile_id, vertices, uvs, indices);
            }
            Atom::AddLineStrip2D {
                id,
                tile_id,
                points,
                width,
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
                    .add_line_strip_2d(id, tile_id, points, width);
            }
            Atom::NewChunk { id } => {
                self.chunks_map.entry(id).or_insert_with(Chunk::default);
            }
            Atom::SetChunk { id, chunk } => {
                // Insert or replace the chunk as-is; caller controls current_chunk separately
                self.chunks_map.insert(id, chunk);
            }
            Atom::RemoveChunk { id } => {
                let was_current = self.current_chunk == Some(id);
                self.chunks_map.remove(&id);
                if was_current {
                    self.current_chunk = None;
                }
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
            Atom::SetRenderMode(m) => {
                self.render_mode = m;
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
            tile_offsets: None,
            tile_counts: None,
            tile_tris: None,
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

    /// Iterate polygons sorted by their chunk's priority (higher first). Allocates a small Vec each call.
    pub fn polys_2d_sorted_by_chunk_priority(&self) -> Vec<(&Poly2D, Option<&AtlasEntry>)> {
        let anim = self.animation_counter as u32;
        let mut pairs: Vec<(&Poly2D, Option<&AtlasEntry>, i32)> = Vec::new();
        for (_, ch) in &self.chunks_map {
            let prio = ch.priority;
            for poly in ch.polys_map.values() {
                let rect = self.frame_rect(&poly.tile_id, anim);
                pairs.push((poly, rect, prio));
            }
        }
        // Sort by priority descending; stable to preserve insertion order within same chunk priority
        pairs.sort_by(|a, b| b.2.cmp(&a.2));
        pairs.into_iter().map(|(p, r, _)| (p, r)).collect()
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
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Compute2DUniforms>() as _,
                        ),
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
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<Compute3DUniforms>() as _,
                        ),
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
            // Mat3 columns (col-major), pad .w = 0.0
            mat2d_c0: [m[(0, 0)], m[(1, 0)], m[(2, 0)], 0.0],
            mat2d_c1: [m[(0, 1)], m[(1, 1)], m[(2, 1)], 0.0],
            mat2d_c2: [m[(0, 2)], m[(1, 2)], m[(2, 2)], 0.0],
            mat2d_inv_c0: [m_inv[(0, 0)], m_inv[(1, 0)], m_inv[(2, 0)], 0.0],
            mat2d_inv_c1: [m_inv[(0, 1)], m_inv[(1, 1)], m_inv[(2, 1)], 0.0],
            mat2d_inv_c2: [m_inv[(0, 2)], m_inv[(1, 2)], m_inv[(2, 2)], 0.0],
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
        for (poly, rect_opt) in self.polys_2d_sorted_by_chunk_priority() {
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
            }
        }

        // --- CPU tiling & binning (8x8 tiles) ---
        let tiles_x = ((fb_w + 7) / 8).max(1);
        let tiles_y = ((fb_h + 7) / 8).max(1);
        let tiles_n = (tiles_x * tiles_y) as usize;
        let mut bins: Vec<Vec<u32>> = vec![Vec::new(); tiles_n];

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
                    bins[idx].push(t as u32);
                }
            }
        }

        // Flatten to offsets/counts/tris
        let mut tile_offsets: Vec<u32> = Vec::with_capacity(tiles_n);
        let mut tile_counts: Vec<u32> = Vec::with_capacity(tiles_n);
        let mut tile_tris: Vec<u32> = Vec::new();
        let mut running: u32 = 0;
        for v in &bins {
            tile_offsets.push(running);
            let c = v.len() as u32;
            tile_counts.push(c);
            running += c;
            tile_tris.extend_from_slice(v);
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
        let m = vek::Mat4::<f32>::identity(); // TODO: store & set via atom later
        let u = Compute3DUniforms {
            background: self.background.into_array(),
            fb_size: [fb_w, fb_h],
            _pad0: [0, 0],
            gp0: self.gp0.into_array(),
            gp1: self.gp1.into_array(),
            gp2: self.gp2.into_array(),
            mat3d_c0: [m[(0, 0)], m[(1, 0)], m[(2, 0)], m[(3, 0)]],
            mat3d_c1: [m[(0, 1)], m[(1, 1)], m[(2, 1)], m[(3, 1)]],
            mat3d_c2: [m[(0, 2)], m[(1, 2)], m[(2, 2)], m[(3, 2)]],
            mat3d_c3: [m[(0, 3)], m[(1, 3)], m[(2, 3)], m[(3, 3)]],
        };
        if let Some(g) = self.gpu.as_ref() {
            queue.write_buffer(g.u3d_buf.as_ref().unwrap(), 0, bytemuck::bytes_of(&u));
        }
        // Ensure atlas is available for sampling on GPU
        self.atlas.ensure_gpu_with(device);
        self.upload_atlas_to_gpu_with(device, queue);
        let view = &surface.gpu.as_ref().unwrap().view;
        let atlas_view = &self.atlas.gpu.as_ref().unwrap().view;
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
            ],
        }));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vm-3d-cs-enc"),
        });
        {
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
