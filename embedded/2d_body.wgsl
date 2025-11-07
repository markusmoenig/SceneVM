// 2D Body. Can be replaced via Atom::SetSource2D

const VM_FLAG_SKIP_CLEAR: u32 = 1u;

@compute @workgroup_size(8,8,1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= U.fb_size.x || py >= U.fb_size.y) { return; }

    let skip_clear = (U.vm_flags & VM_FLAG_SKIP_CLEAR) != 0u;
    if (!skip_clear) {
        // Clear to background if we're the base layer.
        sv_write(px, py, U.background);
    }

    let p = vec2<f32>(f32(px) + 0.5, f32(py) + 0.5);
    let tid = tile_of_px(px, py);
    let ch = sv_shade_tile_pixel(p, px, py, tid);
    if (ch.hit) {
        let mats = textureSampleLevel(atlas_mat_tex, atlas_smp, ch.uv, 0.0);
        let opacity = mats.z;
        let emission = mats.w;

        let base = ch.color;
        let rgb = base.xyz * (1.0 + emission);
        let a   = base.a * opacity;
        sv_write(px, py, vec4<f32>(rgb, a));
    }
}
