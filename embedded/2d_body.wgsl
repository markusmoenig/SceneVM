// 2D Body. Can be replaced via Atom::SetSource2D

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
