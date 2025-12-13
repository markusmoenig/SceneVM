// UI 2D body: shades rounded rects with optional border using atlas/material texels.
// Expects:
// - atlas texel 0: fill color (rgba8)
// - atlas texel 1: border color (rgba8)
// - material texel 0: radius_norm (r), border_norm (g) in 0..1 (normalized to min dimension)

// Uses bindings/types from 2d_header.wgsl (U2D, verts, tile_bins, tile_tris, atlas_tex, atlas_mat_tex, etc.).
// Note: relies on the tiled 2D pipeline bindings already defined in the header; no bindings are re-declared here.

struct Style {
  fill: vec4<f32>,
  border: vec4<f32>,
  radius_norm: f32,
  border_norm: f32,
};

const VM_FLAG_SKIP_CLEAR: u32 = 1u;

fn load_style(tile_index: u32) -> Style {
  let frame = sv_tile_frame(tile_index);
  // Sample centers of texel 0 and 1 in a 2x1 tile.
  let uv_fill = frame.ofs + frame.scale * vec2<f32>(0.25, 0.5);
  let uv_border = frame.ofs + frame.scale * vec2<f32>(0.75, 0.5);
  let uv_params = uv_fill;

  let fill = textureSampleLevel(atlas_tex, atlas_smp, uv_fill, 0.0);
  let border = textureSampleLevel(atlas_tex, atlas_smp, uv_border, 0.0);
  let params = textureSampleLevel(atlas_mat_tex, atlas_smp, uv_params, 0.0);
  let radius_norm = params.r; // 0..1
  let border_norm = params.g; // 0..1
  return Style(fill, border, radius_norm, border_norm);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let px = gid.x;
  let py = gid.y;
  if (px >= U.fb_size.x || py >= U.fb_size.y) { return; }

  // Optionally clear to background like the standard 2D path.
  let skip_clear = (U.vm_flags & VM_FLAG_SKIP_CLEAR) != 0u;
  if (!skip_clear) {
    sv_write(px, py, U.background);
  }

  let tid = tile_of_px(px, py);
  let bins_len = arrayLength(&tile_bins.data);
  if (bins_len == 0u || tid >= bins_len) { return; }
  let bin = tile_bins.data[tid];

  let tris_len = arrayLength(&tile_tris.data);
  let indices_len = arrayLength(&indices.data);
  let verts_len = arrayLength(&verts.data);

  var out_col = vec4<f32>(U.background.rgb, 1.0);
  var covered = false;

  let p = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);

  for (var k: u32 = 0u; k < bin.count; k = k + 1u) {
    let tri_idx = bin.offset + k;
    if (tri_idx >= tris_len) { break; }

    let t = tile_tris.data[tri_idx];
    let base = 3u * t;
    if (base + 2u >= indices_len) { continue; }

    let i0 = indices.data[base + 0u];
    let i1 = indices.data[base + 1u];
    let i2 = indices.data[base + 2u];
    if (i0 >= verts_len || i1 >= verts_len || i2 >= verts_len) { continue; }

    let p0 = verts.data[i0].pos;
    let p1 = verts.data[i1].pos;
    let p2 = verts.data[i2].pos;
    let bh = sv_tri_bary(p, p0, p1, p2);
    if (!bh.hit) { continue; }

    let w = bh.w;
    let uv0 = verts.data[i0].uv;
    let uv1 = verts.data[i1].uv;
    let uv2 = verts.data[i2].uv;
    let uv = w.x * uv0 + w.y * uv1 + w.z * uv2;

    // Sample style (all verts of a tri share the same tile_index)
    let style = load_style(verts.data[i0].tile_index);

    // Local UV inside the quad (assumed normalized 0..1 from vertex UVs)
    let local = uv;

    // Rounded rect SDF in normalized space
    let half = vec2<f32>(0.5, 0.5);
    let r = style.radius_norm;
    let shrink = half - vec2<f32>(r, r);
    let d = abs(local - half) - shrink;
    let dist = length(max(d, vec2<f32>(0.0, 0.0))) + min(max(d.x, d.y), 0.0) - r;

    let border_w = style.border_norm;
    let fw = max(1.0 / max(f32(U.fb_size.x), f32(U.fb_size.y)), 1e-3);
    let body = 1.0 - smoothstep(0.0, fw, dist); // coverage mask 0..1
    let border_band = smoothstep(-border_w - fw, -border_w, dist) - smoothstep(border_w, border_w + fw, dist);

    let fill_col = style.fill;
    let border_col = style.border;
    let surf = mix(border_col, fill_col, border_band);

    let cov = clamp(body, 0.0, 1.0);
    let rgb = mix(U.background.rgb, surf.rgb, cov);
    let a = mix(U.background.a, surf.a, cov);

    out_col = vec4<f32>(rgb, a);
    covered = true;
    break;
  }

  if (covered) {
    sv_write(px, py, out_col);
  }
}
