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

    var base_hit = false;
    var base_color = vec4<f32>(0.0);
    if (ch.hit) {
        let mats = textureSampleLevel(atlas_mat_tex, atlas_smp, ch.uv, 0.0);
        let opacity = mats.z;
        let emission = mats.w;

        let base = ch.color;
        let rgb = base.xyz * (1.0 + emission);
        let a = base.a * opacity;
        base_color = vec4<f32>(rgb, a);
        base_hit = true;
    }

    var dyn_hit = false;
    var dyn_color = vec4<f32>(0.0);
    let dyn_count = scene_data.header.billboard_cmd_count;
    if (dyn_count > 0u) {
        for (var di: u32 = 0u; di < dyn_count; di = di + 1u) {
            let cmd = sd_billboard_cmd(di);
            if (cmd.params.y != DYNAMIC_KIND_BILLBOARD_TILE) {
                continue;
            }
            let bh = sd_billboard_hit_screen(p, cmd);
            if (!bh.hit) { continue; }
            let frame = sv_tile_frame(bh.tile_index);
            let uv = frame.ofs + bh.uv * frame.scale;
            let col = textureSampleLevel(atlas_tex, atlas_smp, uv, 0.0);
            if (col.a < 0.01) { continue; }
            dyn_color = col;
            dyn_hit = true;
            break;
        }
    }

    if (dyn_hit) {
        sv_write(px, py, dyn_color);
        return;
    }

    if (base_hit) {
        sv_write(px, py, base_color);
    }
}
