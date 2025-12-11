struct USdf {
  background: vec4<f32>,
  fb_size: vec2<u32>,
  _pad0: vec2<u32>,
  gp0: vec4<f32>,
  gp1: vec4<f32>,
  gp2: vec4<f32>,
  gp3: vec4<f32>,
  gp4: vec4<f32>,
  gp5: vec4<f32>,
  gp6: vec4<f32>,
  gp7: vec4<f32>,
  gp8: vec4<f32>,
  gp9: vec4<f32>,
  data_len: u32,
  vm_flags: u32,
  anim_counter: u32,
  _pad_tail: u32,
};

@group(0) @binding(0) var<uniform> U: USdf;
@group(0) @binding(1) var color_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> SDF_DATA: array<vec4<f32>>;
@group(0) @binding(3) var atlas_tex: texture_2d<f32>;
@group(0) @binding(4) var atlas_smp: sampler;

fn sdf_data_at(i: u32) -> vec4<f32> {
  let len = max(U.data_len, 1u);
  return SDF_DATA[i % len];
}

fn clear_if_needed(gid: vec3<u32>) {
  // Host can set VM_FLAG_SKIP_CLEAR to keep previous frame alive.
  if (U.vm_flags & 1u) != 0u {
    return;
  }
  textureStore(color_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(U.background.rgb, 1.0));
}

fn sdf_sample_atlas(rect: vec4<f32>, uv: vec2<f32>) -> vec4<f32> {
  let uv_atlas = rect.xy + rect.zw * uv;
  return textureSampleLevel(atlas_tex, atlas_smp, uv_atlas, 0.0);
}
