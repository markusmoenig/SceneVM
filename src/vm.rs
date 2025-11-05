// Non-empty dummy buffers for wgpu STORAGE bindings when a scene grid is empty.
const DUMMY_U32_1: [u32; 1] = [0];
const DUMMY_I32_1: [i32; 1] = [0];

use crate::{Camera3D, CameraKind, Chunk, Light, LightType, Poly2D, Poly3D, Texture};
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
    pub uv_os: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vert3DPod {
    pub pos: [f32; 3],
    pub _pad_pos: f32,     // 16
    pub uv: [f32; 2],      // +8  = 24
    pub _pad_uv: [f32; 2], // +8  = 32  <-- NEW: force 16-byte alignment before next vec4
    pub uv_os: [f32; 4],   // +16 = 48
    pub normal: [f32; 3],
    pub _pad_n: f32, // +16 = 64 total
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

/// VM instruction set
#[derive(Debug)]
pub enum Atom {
    /// Add a tile with `id`, dimensions, and animation frames (RGBA8). Each frame is tightly packed width*height*4 bytes.
    AddTile {
        id: Uuid,
        width: u32,
        height: u32,
        frames: Vec<Vec<u8>>, // frames[f][row*width*4 .. (row+1)*width*4]
        material_frames: Option<Vec<Vec<u8>>>,
    },
    /// Provide or replace per-frame material maps (RGBA = roughness/metallic/opacity/emission) for an existing tile.
    SetTileMaterialFrames {
        id: Uuid,
        frames: Vec<Vec<u8>>,
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
        poly: Poly2D,
    },
    /// Add a 3D polygon (world coords) that references a tile by UUID; indices are local to the chunk.
    AddPoly3D {
        poly: Poly3D,
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
    /// Add a 2D line strip rendered in screen space with a constant pixel width.
    /// Points are in world coordinates; width is in pixels.
    AddLineStrip2Dpx {
        id: GeoId,
        tile_id: Uuid,
        points: Vec<[f32; 2]>,
        width_px: f32,
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
    /// Build/replace the global scene uniform grid over all current 3D geometry
    SetSceneGridCells {
        target_cells: u32,
    },
    /// Reset the scene acceleration structure (will be rebuilt on next BuildSceneGrid)
    ClearSceneGrid,
    /// Set the camera
    SetCamera3D {
        camera: Camera3D,
    },
}

#[derive(Debug, Clone)]
pub struct AtlasEntry {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug)]
struct Tile {
    w: u32,
    h: u32,
    frames: Vec<Vec<u8>>,
    material_frames: Vec<Vec<u8>>,
}

/// Screen-space line strip description (width in pixels; rendered as quads built in screen space).
#[derive(Debug, Clone)]
pub struct LineStrip2D {
    pub id: GeoId,
    pub tile_id: uuid::Uuid,
    pub points: Vec<[f32; 2]>, // world-space points (will be transformed, then rasterized in screen space)
    pub width_px: f32,         // line width in pixels (constant regardless of world scale)
    pub layer: i32,
    pub visible: bool,
    pub material_id: Option<uuid::Uuid>,
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

    // Lights
    pub lights_count: u32,
    _pad_lights_align: [u32; 3],
    _pad_lights_vec3: [u32; 4],

    // Camera3D
    pub cam_pos: [f32; 4], // xyz, pad
    pub cam_fwd: [f32; 4], // xyz, pad
    pub cam_right: [f32; 4],
    pub cam_up: [f32; 4],
    pub cam_vfov_deg: f32,
    pub cam_ortho_half_h: f32,
    pub cam_near: f32,
    pub cam_far: f32,
    pub cam_kind: u32, // 0=OrthoIso, 1=OrbitPersp, 2=FirstPersonPersp
    _pad_cam: [u32; 3],

    pub _pad_tail: [u32; 4],
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Compute2D,
    Compute3D,
}

/// A tiny, CPU-side VM that collects tiles and builds a texture atlas.
/// Packing strategy: simple shelf packer (rows), stable order by insertion.
pub struct VM {
    tiles_map: FxHashMap<Uuid, Tile>,
    tiles_order: Vec<Uuid>,      // insertion order for stable packing
    pub atlas: Texture,          // CPU/GPU-capable atlas texture (albedo)
    pub atlas_material: Texture, // Parallel atlas storing R/M/O/E channels
    atlas_dirty: bool,
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

    pub current_layer: i32,

    // Scene-wide 3D acceleration via grid
    pub scene_grid_cells: u32,
    pub scene_accel: SceneAccel,
    pub accel_dirty: bool,

    // Camera
    pub camera3d: Camera3D,
}

impl VM {
    /// Create a VM with a fixed-size atlas (atlas_w x atlas_h).
    pub fn new(atlas_w: u32, atlas_h: u32) -> Self {
        let mut source2d = String::new();
        if let Some(bytes) = crate::Embedded::get("2d_body.wgsl") {
            if let Ok(source) = std::str::from_utf8(bytes.data.as_ref()) {
                source2d = source.to_string();
            }
        }

        let mut source3d = String::new();
        if let Some(bytes) = crate::Embedded::get("3d_body.wgsl") {
            if let Ok(source) = std::str::from_utf8(bytes.data.as_ref()) {
                source3d = source.to_string();
            }
        }
        Self {
            tiles_map: FxHashMap::default(),
            tiles_order: Vec::new(),
            atlas: Texture::new(atlas_w, atlas_h),
            atlas_material: Texture::new(atlas_w, atlas_h),
            atlas_map: FxHashMap::default(),
            atlas_dirty: true,
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
            source2d,
            source3d,
            transform2d: Mat3::identity(),
            transform3d: Mat4::identity(),
            lights: FxHashMap::default(),
            current_layer: 0,
            scene_accel: SceneAccel::default(),
            accel_dirty: true,
            scene_grid_cells: 5000,
            camera3d: Camera3D::default(),
        }
    }

    #[inline]
    fn default_material_frame(bytes: usize) -> Vec<u8> {
        if bytes == 0 {
            return Vec::new();
        }
        let mut v = Vec::with_capacity(bytes);
        let pixels = bytes / 4;
        for _ in 0..pixels {
            v.extend_from_slice(&[128u8, 0u8, 255u8, 0u8]);
        }
        if v.len() < bytes {
            v.resize(bytes, 0);
        }
        v
    }

    /// Interpret one instruction.
    pub fn execute(&mut self, atom: Atom) {
        match atom {
            Atom::SetGeoVisible { id, visible } => {
                for ch in self.chunks_map.values_mut() {
                    if let Some(p) = ch.polys_map.get_mut(&id) {
                        p.visible = visible;
                    }
                    if let Some(p3_vec) = ch.polys3d_map.get_mut(&id) {
                        for p3 in p3_vec.iter_mut() {
                            p3.visible = visible;
                        }
                        self.accel_dirty = true;
                    }
                }
            }
            Atom::AddTile {
                id,
                width,
                height,
                frames,
                material_frames,
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
                let mut mat_frames = material_frames.unwrap_or_default();
                if mat_frames.is_empty() {
                    mat_frames = (0..frames.len())
                        .map(|_| Self::default_material_frame(need))
                        .collect();
                }
                if mat_frames.len() < frames.len() {
                    let missing = frames.len() - mat_frames.len();
                    mat_frames.extend((0..missing).map(|_| Self::default_material_frame(need)));
                }
                if mat_frames.len() > frames.len() {
                    mat_frames.truncate(frames.len());
                }
                for mf in mat_frames.iter_mut() {
                    if mf.len() < need {
                        mf.resize(need, 0);
                    }
                    if mf.len() > need {
                        mf.truncate(need);
                    }
                }
                if mat_frames.is_empty() {
                    mat_frames.push(Self::default_material_frame(need));
                }
                let is_new = !self.tiles_map.contains_key(&id);
                self.tiles_map.insert(
                    id,
                    Tile {
                        w: width,
                        h: height,
                        frames,
                        material_frames: mat_frames,
                    },
                );
                if is_new {
                    self.tiles_order.push(id);
                }
                self.atlas_dirty = true;
            }
            Atom::AddSolid { id, color } => {
                // Create a 1x1 tile with a single frame of the given color
                let frame = color.to_vec();
                let mat_frame = Self::default_material_frame(4);
                let is_new = !self.tiles_map.contains_key(&id);
                self.tiles_map.insert(
                    id,
                    Tile {
                        w: 1,
                        h: 1,
                        frames: vec![frame],
                        material_frames: vec![mat_frame],
                    },
                );
                if is_new {
                    self.tiles_order.push(id);
                }
                self.atlas_dirty = true;
            }
            Atom::SetTileMaterialFrames { id, frames } => {
                if let Some(tile) = self.tiles_map.get_mut(&id) {
                    let need = (tile.w as usize) * (tile.h as usize) * 4;
                    let mut mats: Vec<Vec<u8>> = frames
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
                    if mats.len() < tile.frames.len() {
                        let missing = tile.frames.len() - mats.len();
                        mats.extend((0..missing).map(|_| Self::default_material_frame(need)));
                    }
                    if mats.len() > tile.frames.len() {
                        mats.truncate(tile.frames.len());
                    }
                    tile.material_frames = mats;
                    self.atlas_dirty = true;
                }
            }
            Atom::BuildAtlas => {
                self.build_atlas();
                self.atlas_dirty = true;
            }
            Atom::AddPoly { poly } => {
                let chunk_id = match self.current_chunk {
                    Some(cid) => cid,
                    None => {
                        let cid = Uuid::new_v4();
                        self.chunks_map.insert(cid, Chunk::default());
                        self.current_chunk = Some(cid);
                        cid
                    }
                };
                self.chunks_map.entry(chunk_id).or_default().add(poly);
            }
            Atom::AddPoly3D { poly } => {
                let chunk_id = match self.current_chunk {
                    Some(cid) => cid,
                    None => {
                        let cid = Uuid::new_v4();
                        self.chunks_map.insert(cid, Chunk::default());
                        self.current_chunk = Some(cid);
                        cid
                    }
                };
                self.chunks_map.entry(chunk_id).or_default().add_3d(poly);
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
            Atom::AddLineStrip2Dpx {
                id,
                tile_id,
                points,
                width_px,
                material_id,
            } => {
                if points.len() < 2 || width_px <= 0.0 {
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
                    .add_line_strip_2d_px(
                        id,
                        tile_id,
                        points,
                        width_px,
                        self.current_layer,
                        material_id,
                    );
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
                self.atlas_material.data.fill(0);
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
                self.atlas_material.data.fill(0);
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
            Atom::SetSceneGridCells { target_cells } => {
                self.scene_grid_cells = target_cells;
                self.accel_dirty = true;
            }
            Atom::ClearSceneGrid => {
                // Reset to an empty 1x1 grid to keep bindings valid
                self.scene_accel = SceneAccel::default();
                self.accel_dirty = true;
            }
            Atom::SetCamera3D { camera } => {
                self.camera3d = camera;
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

        // let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let sampler: wgpu::Sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vm-atlas-sampler-repeat-nearest"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

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
        });
    }

    /// Returns a read-only view of the current color atlas pixels (RGBA8).
    pub fn atlas_pixels(&self) -> &[u8] {
        &self.atlas.data
    }

    /// Returns a read-only view of the material atlas pixels (RGBA8 storing R/M/O/E).
    pub fn material_atlas_pixels(&self) -> &[u8] {
        &self.atlas_material.data
    }

    /// Copies the atlas into a destination pixel slice of size (dst_w x dst_h) RGBA8.
    /// Does not resize the destination; only overlaps are copied line-by-line.
    pub fn copy_atlas_to_slice(&self, dst: &mut [u8], dst_w: u32, dst_h: u32) {
        self.atlas.copy_to_slice(dst, dst_w, dst_h);
    }

    /// Copies the material atlas into a destination pixel slice (RGBA8 R/M/O/E).
    pub fn copy_material_atlas_to_slice(&self, dst: &mut [u8], dst_w: u32, dst_h: u32) {
        self.atlas_material.copy_to_slice(dst, dst_w, dst_h);
    }

    /// Upload the CPU atlas to GPU (creates GPU resources if needed).
    pub fn upload_atlas_to_gpu_with(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.atlas_dirty {
            self.atlas.upload_to_gpu_with(device, queue);
            self.atlas_material.upload_to_gpu_with(device, queue);
            self.atlas_dirty = false;
        }
    }

    /// Download the atlas from GPU into CPU memory; blocks on native, schedules on wasm.
    pub fn download_atlas_from_gpu_with(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.atlas.download_from_gpu_with(device, queue);
        self.atlas_material.download_from_gpu_with(device, queue);
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
        self.atlas_material.data.fill(0);
        self.atlas_map.clear();

        let mut pen_x: u32 = 0;
        let mut pen_y: u32 = 0;
        let mut shelf_h: u32 = 0;

        for id in &self.tiles_order {
            // Copy needed metadata in a short scope to avoid holding an immutable borrow
            let (w, h, frames_len, mat_len) = {
                match self.tiles_map.get(id) {
                    Some(t) => (t.w, t.h, t.frames.len(), t.material_frames.len()),
                    None => continue,
                }
            };
            if w == 0 || h == 0 {
                continue;
            }

            let mut rects: Vec<AtlasEntry> = Vec::with_capacity(frames_len);
            let need_bytes = (w as usize) * (h as usize) * 4;

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
                let (frame_owned, mat_owned) = {
                    let tile = self.tiles_map.get(id).expect("tile must exist");
                    let frame = tile.frames[f].clone();
                    let mat = if f < mat_len {
                        tile.material_frames[f].clone()
                    } else {
                        Self::default_material_frame(need_bytes)
                    };
                    (frame, mat)
                };
                {
                    let atlas_w = self.atlas.width;
                    let dst = &mut self.atlas.data;
                    VM::blit_rgba_into(dst, atlas_w, &frame_owned, w, h, pen_x, pen_y);
                }
                {
                    let atlas_w = self.atlas_material.width;
                    let dst = &mut self.atlas_material.data;
                    VM::blit_rgba_into(dst, atlas_w, &mat_owned, w, h, pen_x, pen_y);
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
                    // material atlas texture
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
                    // material atlas texture (sampled)
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
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
            let mut header_2d = String::new();
            if let Some(bytes) = crate::Embedded::get("2d_header.wgsl") {
                if let Ok(source) = std::str::from_utf8(bytes.data.as_ref()) {
                    header_2d = source.to_string();
                }
            }
            let src2d = [header_2d.as_str(), &self.source2d].concat();
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
            let mut header_3d = String::new();
            if let Some(bytes) = crate::Embedded::get("3d_header.wgsl") {
                if let Ok(source) = std::str::from_utf8(bytes.data.as_ref()) {
                    header_3d = source.to_string();
                }
            }

            let src3d = [header_3d.as_str(), &self.source3d].concat();
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
        if self.atlas_dirty || self.atlas.gpu.is_none() || self.atlas_material.gpu.is_none() {
            self.atlas.ensure_gpu_with(device);
            self.atlas_material.ensure_gpu_with(device);
            self.upload_atlas_to_gpu_with(device, queue);
        }

        // Build transformed 2D geometry (screen-space) and upload to SSBOs
        let mut verts_flat: Vec<Vert2DPod> = Vec::new();
        let mut indices_flat: Vec<u32> = Vec::new();

        // For layer sorting
        #[derive(Clone, Copy)]
        struct TriMeta {
            layer: i32,
            prio: i32,
            ord: u32,
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
                let atlas_w = self.atlas.width as f32;
                let atlas_h = self.atlas.height as f32;
                let ofs_x = rect.x as f32 / atlas_w;
                let ofs_y = rect.y as f32 / atlas_h;
                let scl_x = rect.w as f32 / atlas_w;
                let scl_y = rect.h as f32 / atlas_h;

                let base = verts_flat.len() as u32;

                for (i, v) in poly.vertices.iter().enumerate() {
                    // Apply local and global transforms
                    let local_p = poly.transform * Vec3::new(v[0], v[1], 1.0);
                    let world_p = self.transform2d * local_p;

                    // Keep object UV for GPU-side wrapping
                    let base_uv = poly.uvs[i];

                    verts_flat.push(Vert2DPod {
                        pos: [world_p.x, world_p.y],
                        uv: [base_uv[0], base_uv[1]],
                        uv_os: [ofs_x, ofs_y, scl_x, scl_y],
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
                    });
                    tri_ord = tri_ord.wrapping_add(1);
                }
            }
        }

        // --- Screen-space constant-width lines from chunks (built as quads per segment) ---
        // Transform each point with the same 2D transform used for polys, then expand to a quad
        // using a pixel-space normal so the width is independent of world scale.
        {
            let atlas_w = self.atlas.width as f32;
            let atlas_h = self.atlas.height as f32;

            for (_cid, ch) in &self.chunks_map {
                if ch.lines2d_px.is_empty() {
                    continue;
                }
                for ls in ch.lines2d_px.values() {
                    if !ls.visible {
                        continue;
                    }
                    // Resolve the atlas frame for this tile
                    let rect = match self.frame_rect(&ls.tile_id, self.animation_counter as u32) {
                        Some(r) => r,
                        None => continue,
                    };
                    let ofs_x = rect.x as f32 / atlas_w;
                    let ofs_y = rect.y as f32 / atlas_h;
                    let scl_x = rect.w as f32 / atlas_w;
                    let scl_y = rect.h as f32 / atlas_h;

                    // Precompute full-rect UVs (we'll map along segment length 0..1)
                    let v0v1v2v3 = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];

                    // Transform points to screen space first (same as polys)
                    let mut pts_scr: Vec<[f32; 2]> = Vec::with_capacity(ls.points.len());
                    for p in &ls.points {
                        let local = Vec3::new(p[0], p[1], 1.0);
                        let world = self.transform2d * local;
                        pts_scr.push([world.x, world.y]);
                    }

                    // Emit quads per segment
                    let half = 0.5 * ls.width_px.max(0.0);
                    for seg in 0..(pts_scr.len().saturating_sub(1)) {
                        let p0 = pts_scr[seg];
                        let p1 = pts_scr[seg + 1];
                        let dx = p1[0] - p0[0];
                        let dy = p1[1] - p0[1];
                        let len = (dx * dx + dy * dy).sqrt();
                        if len < 1e-6 {
                            continue;
                        }
                        let nx = -dy / len;
                        let ny = dx / len;
                        let ox = nx * half;
                        let oy = ny * half;

                        // Screen-space quad corners (consistent winding)
                        let q0 = [p0[0] - ox, p0[1] - oy]; // bottom-left
                        let q1 = [p0[0] + ox, p0[1] + oy]; // top-left
                        let q2 = [p1[0] + ox, p1[1] + oy]; // top-right
                        let q3 = [p1[0] - ox, p1[1] - oy]; // bottom-right

                        // Atlas UVs mapped to full rect; U stretched along segment
                        let base = verts_flat.len() as u32;
                        for uv01 in v0v1v2v3 {
                            verts_flat.push(Vert2DPod {
                                pos: [0.0, 0.0],
                                uv: [uv01[0], uv01[1]],
                                uv_os: [ofs_x, ofs_y, scl_x, scl_y],
                            });
                        }
                        // Overwrite positions with the quad
                        let n = verts_flat.len();
                        verts_flat[n - 4].pos = q0;
                        verts_flat[n - 3].pos = q1;
                        verts_flat[n - 2].pos = q2;
                        verts_flat[n - 1].pos = q3;

                        indices_flat.extend_from_slice(&[
                            base + 0,
                            base + 1,
                            base + 2,
                            base + 0,
                            base + 2,
                            base + 3,
                        ]);

                        // Track sorting info: layer from line, prio=0, ord increases
                        tri_meta.push(TriMeta {
                            layer: ls.layer,
                            prio: 0,
                            ord: tri_ord,
                        });
                        tri_ord = tri_ord.wrapping_add(1);
                        tri_meta.push(TriMeta {
                            layer: ls.layer,
                            prio: 0,
                            ord: tri_ord,
                        });
                        tri_ord = tri_ord.wrapping_add(1);
                    }
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
        // Ensure non-zero-sized buffers for binding validation
        let vbytes: Vec<u8> = if verts_flat.is_empty() {
            // one dummy Vert2DPod (pos=0, uv=0) -> 32 bytes
            bytemuck::bytes_of(&Vert2DPod {
                pos: [0.0, 0.0],
                uv: [0.0, 0.0],
                uv_os: [0.0, 0.0, 0.0, 0.0],
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
        let atlas_mat_view = &self.atlas_material.gpu.as_ref().unwrap().view;
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
                    resource: wgpu::BindingResource::TextureView(atlas_mat_view),
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
        let c = self.camera3d;
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
            cam_pos: [c.pos.x, c.pos.y, c.pos.z, 0.0],
            cam_fwd: [c.forward.x, c.forward.y, c.forward.z, 0.0],
            cam_right: [c.right.x, c.right.y, c.right.z, 0.0],
            cam_up: [c.up.x, c.up.y, c.up.z, 0.0],
            cam_vfov_deg: c.vfov_deg,
            cam_ortho_half_h: c.ortho_half_h,
            cam_near: c.near,
            cam_far: c.far,
            cam_kind: match c.kind {
                CameraKind::OrthoIso => 0,
                CameraKind::OrbitPersp => 1,
                CameraKind::FirstPersonPersp => 2,
            },
            _pad_cam: [0, 0, 0],
            _pad_tail: [0, 0, 0, 0],
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

        if self.atlas_dirty || self.atlas.gpu.is_none() || self.atlas_material.gpu.is_none() {
            self.atlas.ensure_gpu_with(device);
            self.atlas_material.ensure_gpu_with(device);
            self.upload_atlas_to_gpu_with(device, queue);
        }

        // --- Build 3D geometry (world space) and upload to SSBOs ---
        let mut v3: Vec<Vert3DPod> = Vec::new();
        let mut i3: Vec<u32> = Vec::new();

        for (_cid, ch) in &self.chunks_map {
            for poly_list in ch.polys3d_map.values() {
                for poly in poly_list {
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

                    let ofs_x = rect.x as f32 / atlas_w;
                    let ofs_y = rect.y as f32 / atlas_h;
                    let scl_x = rect.w as f32 / atlas_w;
                    let scl_y = rect.h as f32 / atlas_h;

                    for (i, p) in poly_pos.iter().enumerate() {
                        let uv0 = poly.uvs[i]; // object-uv per vertex (e.g. 0..1 for a face)
                        let n = poly_nrm[i];
                        v3.push(Vert3DPod {
                            pos: [p[0], p[1], p[2]],
                            _pad_pos: 0.0,
                            uv: [uv0[0], uv0[1]],
                            _pad_uv: [0.0, 0.0],
                            uv_os: [ofs_x, ofs_y, scl_x, scl_y],
                            normal: [n[0], n[1], n[2]],
                            _pad_n: 0.0,
                        });
                    }

                    for &(a, b, c) in &poly.indices {
                        i3.extend_from_slice(&[base + a as u32, base + b as u32, base + c as u32]);
                    }
                }
            }
        }

        if v3.is_empty() {
            v3.push(Vert3DPod {
                pos: [0.0; 3],
                _pad_pos: 0.0,
                uv: [0.0; 2],
                _pad_uv: [0.0, 0.0],
                uv_os: [0.0, 0.0, 0.0, 0.0],
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
        {
            let g = self.gpu.as_mut().unwrap();
            g.v3d_ssbo = Some(v3_buf);
            g.i3d_ssbo = Some(i3_buf);
        }

        // Avoid borrowing self immutably while we need &mut for bind group creation.
        let surface_view = surface.gpu.as_ref().unwrap().view.clone();
        let atlas_view = self.atlas.gpu.as_ref().unwrap().view.clone();
        let atlas_mat_view = self.atlas_material.gpu.as_ref().unwrap().view.clone();

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
                        binding: 11,
                        resource: wgpu::BindingResource::TextureView(&atlas_mat_view),
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
        let pad = 0.1 * diag; // scene padding
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

        // Precompute an epsilon in **world** based on cell size (robustness)
        let cell_eps = cell_size.x.max(cell_size.y).max(cell_size.z) * 1.0;

        // --- 4) Bin triangles into cells with **padded tri AABB** ---

        let tri_count: usize = indices.len() / 3;

        // Parallel version using Rayon (feature-gated); fallback to previous sequential if feature is off.
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;

            // 4a) generate (cell_idx, tri) pairs in parallel
            let mut pairs: Vec<(u32, u32)> = (0..tri_count)
                .into_par_iter()
                .map(|tri| {
                    let i0 = indices[3 * tri + 0] as usize;
                    let i1 = indices[3 * tri + 1] as usize;
                    let i2 = indices[3 * tri + 2] as usize;

                    let p0 = Vec3::new(verts[i0].pos[0], verts[i0].pos[1], verts[i0].pos[2]);
                    let p1 = Vec3::new(verts[i1].pos[0], verts[i1].pos[1], verts[i1].pos[2]);
                    let p2 = Vec3::new(verts[i2].pos[0], verts[i2].pos[1], verts[i2].pos[2]);

                    let mut tmin = vmin(vmin(p0, p1), p2);
                    let mut tmax = vmax(vmax(p0, p1), p2);
                    tmin -= Vec3::broadcast(cell_eps);
                    tmax += Vec3::broadcast(cell_eps);

                    let rel_min = (tmin - bmin) / cell_size;
                    let rel_max = (tmax - bmin) / cell_size;

                    let mut ix0 = rel_min.x.floor() as i32;
                    let mut iy0 = rel_min.y.floor() as i32;
                    let mut iz0 = rel_min.z.floor() as i32;
                    let mut ix1 = rel_max.x.ceil() as i32;
                    let mut iy1 = rel_max.y.ceil() as i32;
                    let mut iz1 = rel_max.z.ceil() as i32;

                    ix0 = ix0.clamp(0, nx as i32 - 1);
                    iy0 = iy0.clamp(0, ny as i32 - 1);
                    iz0 = iz0.clamp(0, nz as i32 - 1);
                    ix1 = ix1.clamp(0, nx as i32 - 1);
                    iy1 = iy1.clamp(0, ny as i32 - 1);
                    iz1 = iz1.clamp(0, nz as i32 - 1);

                    let mut local: Vec<(u32, u32)> = Vec::new();
                    if ix0 <= ix1 && iy0 <= iy1 && iz0 <= iz1 {
                        for z in iz0..=iz1 {
                            for y in iy0..=iy1 {
                                for x in ix0..=ix1 {
                                    let idx = (z as u32 * ny + y as u32) * nx + x as u32;
                                    local.push((idx, tri as u32));
                                }
                            }
                        }
                    }
                    local
                })
                .reduce(
                    || Vec::new(),
                    |mut a, mut b| {
                        a.append(&mut b);
                        a
                    },
                );

            // 4b) sort pairs by cell index (deterministic)
            use rayon::slice::ParallelSliceMut;
            pairs.par_sort_unstable_by_key(|p| p.0);

            // 4c) build CSR with per-cell dedup (deterministic order)
            let cell_count_usize = (nx as usize) * (ny as usize) * (nz as usize);
            let mut offsets = vec![0u32; cell_count_usize];
            let mut counts = vec![0u32; cell_count_usize];
            let mut tris: Vec<u32> = Vec::with_capacity(pairs.len());

            let mut run = 0u32;
            let mut i = 0usize;
            while i < pairs.len() {
                let cell = pairs[i].0 as usize;
                let start = i;
                let key = pairs[i].0;
                // advance i to end of this cell's run
                while i < pairs.len() && pairs[i].0 == key {
                    i += 1;
                }
                // dedup tri ids within this cell
                let mut cell_tris: Vec<u32> = pairs[start..i].iter().map(|&(_, t)| t).collect();
                cell_tris.sort_unstable();
                cell_tris.dedup();

                offsets[cell] = run;
                counts[cell] = cell_tris.len() as u32;
                run += counts[cell];

                tris.extend_from_slice(&cell_tris);
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
            return;
        }

        #[cfg(not(feature = "parallel"))]
        {
            // Storage for CSR
            let cell_count = (nx as usize) * (ny as usize) * (nz as usize);
            let mut cell_vecs: Vec<Vec<u32>> = vec![Vec::new(); cell_count];
            // your existing sequential version (unchanged)
            for tri in 0..tri_count {
                let i0 = indices[3 * tri + 0] as usize;
                let i1 = indices[3 * tri + 1] as usize;
                let i2 = indices[3 * tri + 2] as usize;

                let p0 = Vec3::new(verts[i0].pos[0], verts[i0].pos[1], verts[i0].pos[2]);
                let p1 = Vec3::new(verts[i1].pos[0], verts[i1].pos[1], verts[i1].pos[2]);
                let p2 = Vec3::new(verts[i2].pos[0], verts[i2].pos[1], verts[i2].pos[2]);

                let mut tmin = vmin(vmin(p0, p1), p2);
                let mut tmax = vmax(vmax(p0, p1), p2);
                tmin -= Vec3::broadcast(cell_eps);
                tmax += Vec3::broadcast(cell_eps);

                let rel_min = (tmin - bmin) / cell_size;
                let rel_max = (tmax - bmin) / cell_size;

                let mut ix0 = rel_min.x.floor() as i32;
                let mut iy0 = rel_min.y.floor() as i32;
                let mut iz0 = rel_min.z.floor() as i32;
                let mut ix1 = rel_max.x.ceil() as i32;
                let mut iy1 = rel_max.y.ceil() as i32;
                let mut iz1 = rel_max.z.ceil() as i32;

                ix0 = ix0.clamp(0, nx as i32 - 1);
                iy0 = iy0.clamp(0, ny as i32 - 1);
                iz0 = iz0.clamp(0, nz as i32 - 1);
                ix1 = ix1.clamp(0, nx as i32 - 1);
                iy1 = iy1.clamp(0, ny as i32 - 1);
                iz1 = iz1.clamp(0, nz as i32 - 1);

                if ix0 > ix1 || iy0 > iy1 || iz0 > iz1 {
                    continue;
                }

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
