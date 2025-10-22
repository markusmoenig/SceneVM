use crate::Texture;
use bytemuck::{Pod, Zeroable};
use rustc_hash::FxHashMap;
use uuid::Uuid;
use vek::Vec4;
use vek::{Mat3, Vec3};
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
    AddSolid { id: Uuid, color: [u8; 4] },
    /// Build the atlas for all frames
    BuildAtlas,
    /// Add a polygon (world coords) that references a tile by UUID into the CURRENT chunk; indices are local to the chunk.
    AddPoly {
        id: Uuid,      // polygon id (stable within the chunk)
        tile_id: Uuid, // which tile's frames to sample from
        vertices: Vec<[f32; 2]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<(usize, usize, usize)>,
    },
    /// Create an empty chunk (no switch)
    NewChunk { id: Uuid },
    /// Switch the current chunk (created if missing)
    SetCurrentChunk { id: Uuid },
    /// Set the current animation counter (frame index modulo each tile's frame count)
    SetAnimationCounter(usize),
    /// Set raw 2D compute params (e.g., color or knobs) for prototyping
    SetCompute2DParams(Vec4<f32>),
    /// Set raw 3D compute params (e.g., exposure) for prototyping
    SetCompute3DParams(Vec4<f32>),
    /// Switch between 2D and 3D compute drawing
    SetRenderMode(RenderMode),
    /// Set a 2D transform (Mat3) applied on CPU to polygon vertices before 2D compute draw
    SetTransform2D(Mat3<f32>),
    /// Provide a custom WGSL body for the 2D compute shader. The VM will prepend a header and compile at runtime.
    SetSource2D(String),
    /// Provide a custom WGSL body for the 3D compute shader. The VM will prepend a header and compile at runtime.
    SetSource3D(String),
    /// Clear the atlas and tiles
    ClearAtlas,
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
    pub id: Uuid,
    pub tile_id: Uuid,
    pub vertices: Vec<[f32; 2]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<(usize, usize, usize)>, // triangle list, LOCAL to its chunk (Rusterix-compatible)
    pub transform: Mat3<f32>,                // per-poly local transform
}

#[derive(Debug, Default, Clone)]
pub struct Chunk {
    pub polys_map: FxHashMap<Uuid, Poly2D>,
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
    pub param: [f32; 4],
    pub fb_size: [u32; 2],
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Compute3DUniforms {
    pub param: [f32; 4],
    pub fb_size: [u32; 2],
    _pad: [u32; 2],
}

pub const SCENEVM_2D_CS_WGSL: &str = r#"
struct U2D { param: vec4<f32>, fb_size: vec2<u32>, _pad: vec2<u32>, };
@group(0) @binding(0) var<uniform> U: U2D;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= U.fb_size.x || gid.y >= U.fb_size.y) { return; }
  let uv = vec2<f32>(f32(gid.x)/f32(U.fb_size.x), f32(gid.y)/f32(U.fb_size.y));
  // For now: solid color with simple uv tint; later: raster & lighting
  let col = /*vec4<f32>(U.param.xyz, 1.0); */ vec4<f32>(uv.x, uv.y, 0.0, 1.0);
  textureStore(color_out, vec2<i32>(i32(gid.x), i32(gid.y)), col);
}
"#;

pub const SCENEVM_3D_CS_WGSL: &str = r#"
struct U3D { param: vec4<f32>, fb_size: vec2<u32>, _pad: vec2<u32>, };
@group(0) @binding(0) var<uniform> U: U3D;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= U.fb_size.x || gid.y >= U.fb_size.y) { return; }
  // Placeholder: gradient with param.x as brightness; later we pathtrace here
  let uv = vec2<f32>(f32(gid.x)/f32(U.fb_size.x), f32(gid.y)/f32(U.fb_size.y));
  let b = U.param.x;
  let col = vec4<f32>(uv.x*b, uv.y*b, b, 1.0);
  textureStore(color_out, vec2<i32>(i32(gid.x), i32(gid.y)), col);
}
"#;

pub const SCENEVM_2D_HEADER: &str = r#"
struct U2D { param: vec4<f32>, fb_size: vec2<u32>, _pad: vec2<u32>, };
@group(0) @binding(0) var<uniform> U: U2D;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var atlas_tex: texture_2d<f32>;
@group(0) @binding(3) var atlas_smp: sampler;
struct Vert { pos: vec2<f32>, uv: vec2<f32> };
struct Verts { data: array<Vert> };
struct Indices { data: array<u32> };
@group(0) @binding(4) var<storage, read> verts: Verts;
@group(0) @binding(5) var<storage, read> indices: Indices;

// Helpers
fn sv_write(px: u32, py: u32, c: vec4<f32>) {
  textureStore(color_out, vec2<i32>(i32(px), i32(py)), c);
}
fn sv_sample(uv: vec2<f32>) -> vec4<f32> {
  return textureSampleLevel(atlas_tex, atlas_smp, uv, 0.0);
}

// @compute @workgroup_size(8,8,1)
// fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) { ... }
"#;

pub const SCENEVM_3D_HEADER: &str = r#"
struct U3D { param: vec4<f32>, fb_size: vec2<u32>, _pad: vec2<u32>, };
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

// @compute @workgroup_size(8,8,1)
// fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) { ... }
"#;

pub const DEFAULT_2D_BODY: &str = r#"
@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // bounds check
  if (gid.x >= U.fb_size.x || gid.y >= U.fb_size.y) { return; }

  // pixel center in screen space
  let p = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  let tri_count = arrayLength(&indices.data) / 3u;

  // loop all triangles (MVP)
  for (var t: u32 = 0u; t < tri_count; t = t + 1u) {
    let i0 = indices.data[3u*t + 0u];
    let i1 = indices.data[3u*t + 1u];
    let i2 = indices.data[3u*t + 2u];
    let a = verts.data[i0].pos;
    let b = verts.data[i1].pos;
    let c = verts.data[i2].pos;

    // quick bbox reject
    let minx = min(a.x, min(b.x, c.x));
    let maxx = max(a.x, max(b.x, c.x));
    let miny = min(a.y, min(b.y, c.y));
    let maxy = max(a.y, max(b.y, c.y));
    if (p.x < minx || p.x >= maxx || p.y < miny || p.y >= maxy) { continue; }

    // edge functions
    let e0 = (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
    let e1 = (p.x - b.x) * (c.y - b.y) - (p.y - b.y) * (c.x - b.x);
    let e2 = (p.x - c.x) * (a.y - c.y) - (p.y - c.y) * (a.x - c.x);

    // accept either winding
    if ((e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) || (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)) {
      // Compute barycentrics using triangle area
      let area = abs((b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x));
      if (area > 0.0) { // guard degenerate
        let w0 = abs((b.x - p.x)*(c.y - p.y) - (b.y - p.y)*(c.x - p.x)) / area;
        let w1 = abs((c.x - p.x)*(a.y - p.y) - (c.y - p.y)*(a.x - p.x)) / area;
        let w2 = 1.0 - w0 - w1;

        // Interpolate UVs and sample
        let uv0 = verts.data[i0].uv;
        let uv1 = verts.data[i1].uv;
        let uv2 = verts.data[i2].uv;
        let uv = uv0 * w0 + uv1 * w1 + uv2 * w2;

        let col = sv_sample(uv);
        sv_write(gid.x, gid.y, col);
        return; // early-out once a covering tri is drawn
      }
    }
  }

  // No triangle covered this pixel → leave as-is (or clear below)
  // sv_write(gid.x, gid.y, vec4<f32>(0.0, 0.0, 0.0, 1.0));
}
"#;

pub const DEFAULT_3D_BODY: &str = r#"
@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= U.fb_size.x || gid.y >= U.fb_size.y) { return; }
  let uv = vec2<f32>(f32(gid.x)/f32(U.fb_size.x), f32(gid.y)/f32(U.fb_size.y));
  let b = U.param.x;
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
    // --- Compute pipeline params
    pub compute2d_params: Vec4<f32>,
    pub compute3d_params: Vec4<f32>,
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
            compute2d_params: Vec4::new(1.0, 0.8, 0.2, 1.0),
            compute3d_params: Vec4::new(1.0, 1.0, 1.0, 1.0),
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
                // Ensure we have a current chunk; if not, create one implicitly.
                debug_assert_eq!(vertices.len(), uvs.len(), "vertices/uvs length mismatch");
                #[cfg(debug_assertions)]
                {
                    let vlen = vertices.len();
                    for &(a, b, c) in &indices {
                        assert!(a < vlen && b < vlen && c < vlen, "index out of range");
                    }
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
                let poly = Poly2D {
                    id,
                    tile_id,
                    vertices,
                    uvs,
                    indices,
                    transform: Mat3::identity(),
                };
                self.chunks_map
                    .entry(chunk_id)
                    .or_default()
                    .polys_map
                    .insert(id, poly);
            }
            Atom::NewChunk { id } => {
                self.chunks_map.entry(id).or_insert_with(Chunk::default);
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
            Atom::ClearAtlas => {
                self.atlas_map.clear();
                self.tiles_map.clear();
                self.tiles_order.clear();
                self.atlas.data.fill(0);
                self.chunks_map.clear();
                self.current_chunk = None;
                self.animation_counter = 0;
                self.compute2d_params = Vec4::new(1.0, 0.8, 0.2, 1.0);
                self.compute3d_params = Vec4::new(1.0, 1.0, 1.0, 1.0);
                self.render_mode = RenderMode::Compute2D;
            }
            Atom::SetCompute2DParams(v) => {
                self.compute2d_params = v;
            }
            Atom::SetCompute3DParams(v) => {
                self.compute3d_params = v;
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

    /// Iterate polygons ready for drawing (either current chunk or all chunks if none selected): (poly, atlas rect)
    pub fn polys_2d(&self) -> impl Iterator<Item = (&Poly2D, Option<&AtlasEntry>)> {
        let anim = self.animation_counter as u32;
        // Gather iterators depending on chunk selection
        let it: Box<dyn Iterator<Item = &Poly2D> + '_> = if let Some(cid) = self.current_chunk {
            if let Some(ch) = self.chunks_map.get(&cid) {
                Box::new(ch.polys_map.values())
            } else {
                Box::new(std::iter::empty::<&Poly2D>())
            }
        } else {
            Box::new(
                self.chunks_map
                    .values()
                    .flat_map(|ch| ch.polys_map.values()),
            )
        };
        it.map(move |p| {
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
        let u = Compute2DUniforms {
            param: self.compute2d_params.into_array(),
            fb_size: [fb_w, fb_h],
            _pad: [0, 0],
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
        for (poly, rect_opt) in self.polys_2d() {
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
        let u = Compute3DUniforms {
            param: self.compute3d_params.into_array(),
            fb_size: [fb_w, fb_h],
            _pad: [0, 0],
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
