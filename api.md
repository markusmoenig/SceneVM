# SceneVM API Documentation

## Overview

SceneVM provides a powerful shader-based rendering system with separate header and body shaders. The **header shaders** contain core rendering utilities and cannot be replaced, while **body shaders** can be fully customized via the API.

## Naming Conventions

### Prefix System

- **`sv_`** - SceneVM Core Functions: Core rendering utilities provided by SceneVM
- **`sd_`** - Scene Data Functions: Functions that access scene data structures  
- **No Prefix** - Helper functions and data structures

### File Structure

- **`2d_header.wgsl`** / **`3d_header.wgsl`** - Core utilities (immutable)
- **`2d_body.wgsl`** / **`3d_body.wgsl`** - Customizable rendering logic

## 2D API Reference

### Core Rendering Functions

#### `sv_write(px: u32, py: u32, c: vec4<f32>)`
Writes a pixel color to the framebuffer at the specified coordinates.

**Parameters:**
- `px`, `py`: Pixel coordinates (0-based)
- `c`: RGBA color to write

#### `sv_sample(uv: vec2<f32>) -> vec4<f32>`
Samples the atlas texture at the given UV coordinates.

**Parameters:**
- `uv`: Texture coordinates [0,1]
- **Returns:** Sampled RGBA color

#### `sv_shade_tile_pixel(p: vec2<f32>, px: u32, py: u32, tid: u32) -> ColorHit`
Main 2D pixel shading function that tests triangles in a tile.

**Parameters:**
- `p`: Screen-space position
- `px`, `py`: Pixel coordinates
- `tid`: Tile index
- **Returns:** `ColorHit` structure with hit information

### Geometry Functions

#### `sv_tri_bary(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> BaryHit`
Computes barycentric coordinates for a point relative to a triangle.

**Parameters:**
- `p`: Test point
- `a`, `b`, `c`: Triangle vertices
- **Returns:** `BaryHit` with barycentric weights

#### `sv_tri_color(p: vec2<f32>, i0: u32, i1: u32, i2: u32) -> ColorHit`
Computes the color for a triangle at a given screen position.

**Parameters:**
- `p`: Screen position
- `i0`, `i1`, `i2`: Vertex indices
- **Returns:** `ColorHit` with color and hit status

#### `sv_edge(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32`
Computes the signed edge function value.

#### `sv_min_edge_distance_px(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32`
Computes the minimum distance to triangle edges in pixels.

### Coordinate Transformation

#### `sv_screen_from_world(world: vec3<f32>) -> vec2<f32>`
Transforms 3D world coordinates to 2D screen coordinates.

#### `sv_world_from_screen(pix: vec2<f32>) -> vec2<f32>`
Transforms 2D screen coordinates to world coordinates.

### Animation & Texture Functions

#### `sv_tile_frame(tile_index: u32) -> TileFrame`
Gets the current frame data for an animated tile.

**Parameters:**
- `tile_index`: Tile identifier
- **Returns:** `TileFrame` with offset and scale

#### `sv_tri_atlas_uv(i0: u32, i1: u32, i2: u32, w: vec3<f32>) -> vec2<f32>`
Computes atlas UV coordinates from triangle vertices and barycentric weights.

### Random Number Generation

#### `sv_seed(px: u32, py: u32, salt: u32) -> u32`
Generates a seed for random number generation.

#### `sv_rand01(seed: u32) -> f32`
Generates a random float in [0,1) from a seed.

### 2D Material Support
**Note:** 2D does not have dedicated material sampling functions. Materials must be sampled manually:

```wgsl
// Manual material sampling in 2D body shaders
let mats = textureSampleLevel(atlas_mat_tex, atlas_smp, uv, 0.0);
let opacity = mats.z;    // Material opacity
let emission = mats.w;   // Material emission
```

## 3D API Reference

### Core Rendering Functions

#### `sv_trace_grid(ro: vec3<f32>, rd: vec3<f32>, tmin: f32, tmax: f32) -> TraceHit`
Performs grid-accelerated ray tracing through the scene.

**Parameters:**
- `ro`: Ray origin
- `rd`: Ray direction (normalized)
- `tmin`, `tmax`: Ray segment bounds
- **Returns:** `TraceHit` with intersection data

#### `sv_ray_tri_full(ro: vec3<f32>, rd: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> Hit3DFull`
Performs ray-triangle intersection test.

**Parameters:**
- `ro`, `rd`: Ray origin and direction
- `a`, `b`, `c`: Triangle vertices
- **Returns:** `Hit3DFull` with intersection details

#### `sv_grid_active() -> bool`
Checks if grid acceleration is enabled.

### Texture & Material Functions

#### `sv_tri_sample_albedo(i0: u32, i1: u32, i2: u32, bu: f32, bv: f32) -> vec4<f32>`
Samples albedo texture for a triangle using barycentric coordinates.

#### `sv_tri_sample_rmoe(i0: u32, i1: u32, i2: u32, bu: f32, bv: f32) -> vec4<f32>`
Samples material properties (Roughness/Metallic/Opacity/Emission).

**Material Channel Mapping:**
- `x`: Roughness
- `y`: Metallic  
- `z`: Opacity
- `w`: Emission

#### `sv_tri_atlas_uv_obj(i0: u32, i1: u32, i2: u32, bu: f32, bv: f32) -> vec2<f32>`
Computes atlas UV coordinates from object-space UVs.

### Geometry & Lighting

#### `sv_tri_tbn(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, uv0: vec2<f32>, uv1: vec2<f32>, uv2: vec2<f32>) -> mat3x3<f32>`
Computes tangent-bitangent-normal matrix for a triangle.

#### `sv_luma(rgb: vec3<f32>) -> f32`
Computes luminance from RGB color.

#### `sv_interp3(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, u: f32, v: f32) -> vec3<f32>`
Interpolates between three vectors using barycentric coordinates.

### Camera Functions

#### `cam_ray(uv: vec2<f32>) -> Ray`
Generates a camera ray for the given screen UV coordinates.

**Parameters:**
- `uv`: Normalized screen coordinates [0,1]
- **Returns:** `Ray` with origin and direction

## Scene Data API (sd_ functions)

### Data Access Functions

#### `sd_data_word(idx: u32) -> u32`
Reads a 32-bit word from scene data.

#### `sd_vec4u(base_word: u32) -> vec4<u32>`
Reads a vector4 of unsigned integers from scene data.

#### `sd_vec4f(base_word: u32) -> vec4<f32>`
Reads a vector4 of floats from scene data.

### Lighting Functions

#### `sd_light(li: u32) -> LightWGSL`
Retrieves light data from the scene.

**Parameters:**
- `li`: Light index
- **Returns:** `LightWGSL` structure with light properties

### Billboard Functions

#### `sd_billboard_cmd(idx: u32) -> DynBillboardCmd`
Retrieves billboard command data.

#### `sd_billboard_hit_screen(pix: vec2<f32>, cmd: DynBillboardCmd) -> DynBillboardHit2D`
Tests 2D screen-space billboard intersection.

#### `sd_ray_billboard(ro: vec3<f32>, rd: vec3<f32>, cmd: DynBillboardCmd) -> DynBillboardHit`
Tests 3D ray-billboard intersection.

## Data Structures

### 2D Structures

```wgsl
struct ColorHit {
    hit: bool,
    color: vec4<f32>,
    tri: u32,
    uv: vec2<f32>
}

struct BaryHit {
    hit: bool,
    w: vec3<f32>
}

struct DynBillboardHit2D {
    hit: bool,
    uv: vec2<f32>,
    tile_index: u32
}
```

### 3D Structures

```wgsl
struct Hit3DFull {
    hit: bool,
    t: f32,
    u: f32,
    v: f32,
    Ng: vec3<f32>
}

struct TraceHit {
    hit: bool,
    t: f32,
    tri: u32,
    u: f32,
    v: f32,
    Ng: vec3<f32>
}

struct Ray {
    ro: vec3<f32>,
    rd: vec3<f32>
}

struct LightWGSL {
    header: vec4<u32>,    // [light_type, emitting, _, _]
    position: vec4<f32>,  // xyz, _
    color: vec4<f32>,     // rgb, _
    params0: vec4<f32>,   // [intensity, radius, startD, endD]
    params1: vec4<f32>    // [flicker, _, _, _]
}
```

### Animation Structures

```wgsl
struct TileFrame {
    ofs: vec2<f32>,
    scale: vec2<f32>
}

struct TileAnimMeta {
    first_frame: u32,
    frame_count: u32
}
```

## Uniform Structures

### 2D Uniforms (U2D)
```wgsl
struct U2D {
    background: vec4<f32>,
    fb_size: vec2<u32>,
    gp0-gp9: vec4<f32>[10],  // General purpose parameters
    mat2d_c0-c2: vec4<f32>,  // 2D transformation matrix
    mat2d_inv_c0-c2: vec4<f32>, // Inverse transformation
    lights_count: u32,
    vm_flags: u32,
    anim_counter: u32
}
```

### 3D Uniforms (U3D)
```wgsl
struct U3D {
    background: vec4<f32>,
    fb_size: vec2<u32>,
    gp0-gp9: vec4<f32>[10],  // General purpose parameters
    mat3d_c0-c3: vec4<f32>,  // 3D transformation matrix
    lights_count: u32,
    vm_flags: u32,
    anim_counter: u32,
    cam_pos: vec4<f32>,      // Camera position
    cam_fwd: vec4<f32>,      // Camera forward
    cam_right: vec4<f32>,    // Camera right
    cam_up: vec4<f32>,       // Camera up
    cam_vfov_deg: f32,       // Vertical FOV (degrees)
    cam_ortho_half_h: f32,   // Orthographic half-height
    cam_near: f32,           // Near plane
    cam_far: f32,            // Far plane
    cam_kind: u32            // Camera type
}
```

## Usage Examples

### Custom 2D Body Shader

```wgsl
// Custom 2D rendering with manual material sampling
@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x; let py = gid.y;
    if (px >= U.fb_size.x || py >= U.fb_size.y) { return; }
    
    // Use SceneVM utilities
    let p = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);
    let tid = tile_of_px(px, py);
    let ch = sv_shade_tile_pixel(p, px, py, tid);
    
    if (ch.hit) {
        // Manual material sampling
        let mats = textureSampleLevel(atlas_mat_tex, atlas_smp, ch.uv, 0.0);
        let opacity = mats.z;
        let emission = mats.w;
        
        // Custom shading with materials
        let base = ch.color;
        let rgb = base.xyz * (1.0 + emission);
        let a = base.a * opacity;
        
        // Apply custom tint
        let final_color = vec4<f32>(rgb * vec3<f32>(1.0, 0.8, 0.8), a);
        sv_write(px, py, final_color);
    }
}
```

### Custom 3D Body Shader

```wgsl
// Custom 3D rendering with advanced materials
@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x; let py = gid.y;
    if (px >= U.fb_size.x || py >= U.fb_size.y) { return; }
    
    let cam_uv = vec2<f32>((f32(px) + 0.5) / f32(U.fb_size.x),
                           (f32(py) + 0.5) / f32(U.fb_size.y));
    let ray = cam_ray(cam_uv);
    
    // Use grid acceleration
    let hit = sv_trace_grid(ray.ro, normalize(ray.rd), 0.001, 1000.0);
    
    if (hit.hit) {
        // Get triangle indices
        let i0 = indices3d.data[3u*hit.tri + 0u];
        let i1 = indices3d.data[3u*hit.tri + 1u];
        let i2 = indices3d.data[3u*hit.tri + 2u];
        
        // Sample materials using built-in functions
        let albedo = sv_tri_sample_albedo(i0, i1, i2, hit.u, hit.v);
        let mats = sv_tri_sample_rmoe(i0, i1, i2, hit.u, hit.v);
        
        // Custom material processing
        let roughness = mats.x;
        let metallic = mats.y;
        let opacity = mats.z;
        let emission = mats.w;
        
        // Custom lighting calculation
        let P = ray.ro + ray.rd * hit.t;
        let N = hit.Ng;
        
        // Simple custom shading
        let lit_color = albedo.xyz * (0.1 + 0.9 * max(dot(N, normalize(vec3<f32>(1.0, 1.0, 1.0))), 0.0));
        let final_color = vec4<f32>(lit_color + emission * albedo.xyz, opacity);
        
        sv_write(px, py, final_color);
    }
}
```

### Advanced: Procedural Effects with Random Functions

```wgsl
// Procedural dithering effect
@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x; let py = gid.y;
    if (px >= U.fb_size.x || py >= U.fb_size.y) { return; }
    
    let seed = sv_seed(px, py, U.anim_counter);
    let noise = sv_rand01(seed);
    
    // Apply dithering to existing content
    let existing_color = textureLoad(color_out, vec2<i32>(i32(px), i32(py)), 0);
    let dithered = existing_color + vec4<f32>(noise - 0.5) * 0.1;
    
    sv_write(px, py, dithered);
}
```

## Best Practices

### Performance Guidelines
1. **Use grid acceleration** for complex 3D scenes with `sv_trace_grid()`
2. **Leverage tile-based rendering** in 2D with `sv_shade_tile_pixel()`
3. **Batch material sampling** when possible to reduce texture lookups
4. **Use built-in functions** instead of reimplementing core algorithms

### API Usage
1. **Always use `sv_` functions** for core rendering operations
2. **Access scene data safely** through `sd_` functions with bounds checking
3. **Handle material differences** between 2D (manual) and 3D (built-in)
4. **Use coordinate systems correctly** with transformation functions

### Custom Shader Development
1. **Start with the default body shaders** as templates
2. **Test with simple modifications** before complex changes
3. **Use the random functions** for procedural effects and anti-aliasing
4. **Leverage animation system** with `sv_tile_frame()` for dynamic content

## Constants Reference

### System Constants
- `VM_FLAG_SKIP_CLEAR = 1u` - Skip background clearing flag
- `DYNAMIC_KIND_BILLBOARD_TILE = 0u` - Billboard tile type identifier

### Data Structure Sizes
- `SCENE_LIGHT_WORDS = 20u` - Light data structure size in words
- `SCENE_BILLBOARD_CMD_WORDS = 16u` - Billboard command size in words

### Camera Types
- `0` - OrthoIso (Orthographic Isometric)
- `1` - OrbitPersp (Orbiting Perspective)  
- `2` - FirstPersonPersp (First Person Perspective)

## Troubleshooting

### Common Issues
1. **Missing triangles**: Ensure grid acceleration is properly configured
2. **Material artifacts**: Check UV coordinates and atlas mappings
3. **Performance problems**: Use grid acceleration for complex scenes
4. **Coordinate confusion**: Use transformation functions consistently

### Debugging Tips
1. **Use barycentric debug colors** to visualize triangle coverage
2. **Test with simple scenes** before complex ones
3. **Check uniform values** for correct camera and transformation settings
4. **Verify texture bindings** for atlas and material textures

This API provides a robust foundation for creating custom rendering pipelines while maintaining performance and compatibility with SceneVM's core systems.