pub mod atlas;
pub mod bbox2d;
pub mod camera3d;
pub mod chunk;
pub mod intodata;
pub mod light;
pub mod poly2d;
pub mod poly3d;
pub mod texture;
pub mod vm;

use rust_embed::RustEmbed;
#[derive(RustEmbed)]
#[folder = "embedded/"]
#[exclude = "*.txt"]
#[exclude = "*.DS_Store"]
pub struct Embedded;

pub use crate::{
    atlas::{AtlasEntry, SharedAtlas},
    bbox2d::BBox2D,
    camera3d::{Camera3D, CameraKind},
    chunk::Chunk,
    intodata::IntoDataInput,
    light::{Light, LightType},
    poly2d::Poly2D,
    poly3d::Poly3D,
    texture::Texture,
    vm::{Atom, GeoId, LineStrip2D, RenderMode, VM},
};

use image;
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::OnceLock;
#[cfg(target_arch = "wasm32")]
use std::{cell::Cell, future::Future, rc::Rc};
#[cfg(target_arch = "wasm32")]
use std::{
    pin::Pin,
    task::{Context, Poll},
};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::spawn_local;

/// Result of a call to `render_frame`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum RenderResult {
    /// We copied pixels to the caller's buffer this call (may still have a new frame in flight on WASM)
    Presented,
    /// On WASM: GPU init not finished; nothing rendered yet.
    InitPending,
    /// On WASM: a GPU readback is in flight; we presented the last completed frame this call.
    ReadbackPending,
}

pub struct GPUState {
    _instance: wgpu::Instance,
    _adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// Main render surface for SceneVM
    surface: Texture,
}

#[allow(dead_code)]
#[derive(Clone)]
struct GlobalGpu {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[allow(dead_code)]
#[cfg(not(target_arch = "wasm32"))]
static GLOBAL_GPU: OnceLock<GlobalGpu> = OnceLock::new();

#[cfg(target_arch = "wasm32")]
thread_local! {
    static GLOBAL_GPU_WASM: RefCell<Option<GlobalGpu>> = RefCell::new(None);
}

// --- WASM async map flag future support ---
#[cfg(target_arch = "wasm32")]
struct MapReadyFuture {
    flag: Rc<Cell<bool>>,
}

#[cfg(target_arch = "wasm32")]
impl Future for MapReadyFuture {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.flag.get() {
            Poll::Ready(())
        } else {
            // Re-schedule ourselves to be polled again soon.
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

pub struct SceneVM {
    /// The intended render target size; used by either backend.
    size: (u32, u32),

    /// When `Some`, GPU rendering is enabled and initialized; otherwise CPU path.
    gpu: Option<GPUState>,
    #[cfg(target_arch = "wasm32")]
    needs_gpu_init: bool,
    #[cfg(target_arch = "wasm32")]
    init_in_flight: bool,

    atlas: SharedAtlas,
    pub vm: VM,
    overlay_vms: Vec<VM>,
    active_vm_index: usize,
    log_layer_activity: bool,
}

impl Default for SceneVM {
    fn default() -> Self {
        Self::new(100, 100)
    }
}

impl SceneVM {
    fn refresh_layer_metadata(&mut self) {
        self.vm.set_layer_index(0);
        self.vm.set_activity_logging(self.log_layer_activity);
        for (i, vm) in self.overlay_vms.iter_mut().enumerate() {
            vm.set_layer_index(i + 1);
            vm.set_activity_logging(self.log_layer_activity);
        }
    }

    fn total_vm_count(&self) -> usize {
        1 + self.overlay_vms.len()
    }

    fn vm_ref_by_index(&self, index: usize) -> Option<&VM> {
        if index == 0 {
            Some(&self.vm)
        } else {
            self.overlay_vms.get(index.saturating_sub(1))
        }
    }

    fn vm_mut_by_index(&mut self, index: usize) -> Option<&mut VM> {
        if index == 0 {
            Some(&mut self.vm)
        } else {
            self.overlay_vms.get_mut(index.saturating_sub(1))
        }
    }

    fn draw_all_vms(
        base_vm: &mut VM,
        overlays: &mut [VM],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: &mut Texture,
        w: u32,
        h: u32,
    ) {
        base_vm.draw_into(device, queue, surface, w, h);
        for vm in overlays {
            vm.draw_into(device, queue, surface, w, h);
        }
    }

    /// Total number of VM layers (base + overlays).
    pub fn vm_layer_count(&self) -> usize {
        self.total_vm_count()
    }

    /// Append a new VM layer that will render on top of the existing ones. Returns its layer index.
    pub fn add_vm_layer(&mut self) -> usize {
        let mut vm = VM::new_with_shared_atlas(self.atlas.clone());
        vm.set_skip_surface_clear(true);
        self.overlay_vms.push(vm);
        self.refresh_layer_metadata();
        self.total_vm_count() - 1
    }

    /// Remove a VM layer by index (cannot remove the base layer at index 0).
    pub fn remove_vm_layer(&mut self, index: usize) -> Option<VM> {
        if index == 0 {
            return None;
        }
        let idx = index - 1;
        if idx >= self.overlay_vms.len() {
            return None;
        }
        let removed = self.overlay_vms.remove(idx);
        if self.active_vm_index >= self.total_vm_count() {
            self.active_vm_index = self.total_vm_count().saturating_sub(1);
        }
        self.refresh_layer_metadata();
        Some(removed)
    }

    /// Switch the VM layer targeted by `execute`. Returns `true` if the index existed.
    pub fn set_active_vm(&mut self, index: usize) -> bool {
        if index < self.total_vm_count() {
            self.active_vm_index = index;
            true
        } else {
            false
        }
    }

    /// Index of the currently active VM used by `execute`.
    pub fn active_vm_index(&self) -> usize {
        self.active_vm_index
    }

    /// Enable or disable drawing for a VM layer. Disabled layers still receive commands.
    pub fn set_layer_enabled(&mut self, index: usize, enabled: bool) -> bool {
        if let Some(vm) = self.vm_mut_by_index(index) {
            vm.set_enabled(enabled);
            true
        } else {
            false
        }
    }

    /// Toggle verbose per-layer logging for uploads/atlas/grid events.
    pub fn set_layer_activity_logging(&mut self, enabled: bool) {
        self.log_layer_activity = enabled;
        self.refresh_layer_metadata();
    }

    /// Borrow the currently active VM immutably.
    pub fn active_vm(&self) -> &VM {
        self.vm_ref_by_index(self.active_vm_index)
            .expect("active VM index out of range")
    }

    /// Borrow the currently active VM mutably.
    pub fn active_vm_mut(&mut self) -> &mut VM {
        self.vm_mut_by_index(self.active_vm_index)
            .expect("active VM index out of range")
    }

    /// Prints statistics about 2D and 3D polygons currently loaded in all chunks.
    pub fn print_geometry_stats(&self) {
        let mut total_2d = 0usize;
        let mut total_3d = 0usize;
        let mut total_lines = 0usize;

        for vm in std::iter::once(&self.vm).chain(self.overlay_vms.iter()) {
            for (_cid, ch) in &vm.chunks_map {
                total_2d += ch.polys_map.len();
                total_3d += ch.polys3d_map.values().map(|v| v.len()).sum::<usize>();
                total_lines += ch.lines2d_px.len();
            }
        }

        println!(
            "[SceneVM] Geometry Stats → 2D polys: {} | 3D polys: {} | 2D lines: {} | Total: {}",
            total_2d,
            total_3d,
            total_lines,
            total_2d + total_3d + total_lines
        );
    }

    /// Executes a single atom on the currently active VM layer.
    pub fn execute(&mut self, atom: Atom) {
        let affects_atlas = SceneVM::atom_touches_atlas(&atom);
        let active = self.active_vm_index;
        if active == 0 {
            self.vm.execute(atom);
        } else if let Some(vm) = self.vm_mut_by_index(active) {
            vm.execute(atom);
        }
        if affects_atlas {
            self.for_each_vm_mut(|vm| vm.mark_all_geometry_dirty());
        }
    }

    /// Is the GPU initialized and ready?
    pub fn is_gpu_ready(&self) -> bool {
        if self.gpu.is_some() {
            #[cfg(target_arch = "wasm32")]
            {
                return !self.needs_gpu_init && !self.init_in_flight;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                return true;
            }
        }
        false
    }

    /// Is a GPU readback currently in flight (WASM only)? Always false on native.
    pub fn frame_in_flight(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(gpu) = &self.gpu {
                return gpu
                    .surface
                    .gpu
                    .as_ref()
                    .and_then(|g| g.map_ready.as_ref())
                    .is_some();
            }
            return false;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }
    /// Create a new SceneVM. Always uses GPU backend.
    pub fn new(initial_width: u32, initial_height: u32) -> Self {
        #[cfg(target_arch = "wasm32")]
        {
            let atlas = SharedAtlas::new(4096, 4096);
            let mut this = Self {
                size: (initial_width, initial_height),
                gpu: None,
                needs_gpu_init: true,
                init_in_flight: false,
                atlas: atlas.clone(),
                vm: VM::new_with_shared_atlas(atlas.clone()),
                overlay_vms: Vec::new(),
                active_vm_index: 0,
                log_layer_activity: false,
            };
            this.refresh_layer_metadata();
            this
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: { wgpu::Backends::all() },
                ..Default::default()
            });
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                }))
                .expect("No compatible GPU adapter found");

            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("scenevm-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }))
                .expect("Failed to create wgpu device");

            let mut surface = Texture::new(initial_width, initial_height);
            surface.ensure_gpu_with(&device);

            let gpu = GPUState {
                _instance: instance,
                _adapter: adapter,
                device,
                queue,
                surface,
            };

            let atlas = SharedAtlas::new(4096, 4096);
            let mut this = Self {
                size: (initial_width, initial_height),
                gpu: Some(gpu),
                atlas: atlas.clone(),
                vm: VM::new_with_shared_atlas(atlas.clone()),
                overlay_vms: Vec::new(),
                active_vm_index: 0,
                log_layer_activity: false,
            };
            this.refresh_layer_metadata();
            this
        }
    }

    /// Initialize GPU backend asynchronously on WASM. On native, this will initialize synchronously if not already.
    pub async fn init_async(&mut self) {
        // If already initialized, nothing to do.
        if self.gpu.is_some() {
            return;
        }

        #[cfg(target_arch = "wasm32")]
        {
            if !self.needs_gpu_init {
                return;
            }
            if global_gpu_get().is_none() {
                global_gpu_init_async().await;
            }
            let gg = global_gpu_get().expect("Global GPU not initialized");
            let (w, h) = self.size;
            let mut surface = Texture::new(w, h);
            surface.ensure_gpu_with(&gg.device);
            let gpu = GPUState {
                _instance: gg.instance,
                _adapter: gg.adapter,
                device: gg.device,
                queue: gg.queue,
                surface,
            };
            self.gpu = Some(gpu);
            self.needs_gpu_init = false;
            #[cfg(debug_assertions)]
            {
                web_sys::console::log_1(&"SceneVM WebGPU initialized (global)".into());
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.gpu.is_some() {
                return;
            }
            let (w, h) = self.size;
            let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                backends: { wgpu::Backends::all() },
                ..Default::default()
            });
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                }))
                .expect("No compatible GPU adapter found");

            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: Some("scenevm-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                }))
                .expect("Failed to create wgpu device");

            let mut surface = Texture::new(w, h);
            surface.ensure_gpu_with(&device);

            let gpu = GPUState {
                _instance: instance,
                _adapter: adapter,
                device,
                queue,
                surface,
            };
            self.gpu = Some(gpu);
        }
    }

    /// Blit a `Texture` via GPU to the main surface texture, if GPU is ready.
    pub fn blit_texture(
        &mut self,
        tex: &mut Texture,
        _cpu_pixels: &mut [u8],
        _buf_w: u32,
        _buf_h: u32,
    ) {
        if let Some(g) = self.gpu.as_ref() {
            tex.gpu_blit_to_storage(g, &g.surface.gpu.as_ref().unwrap().texture);
        }
    }

    /// Draw: if GPU is present, run the compute path. Returns immediately if GPU is not yet ready (WASM before init).
    #[cfg(not(target_arch = "wasm32"))]
    fn draw(&mut self, out_pixels: &mut [u8], out_w: u32, out_h: u32) {
        // GPU-only: do nothing if GPU is not ready (e.g., WASM before init)
        let (gpu_slot, base_vm, overlays) = (&mut self.gpu, &mut self.vm, &mut self.overlay_vms);
        let Some(gpu) = gpu_slot.as_mut() else {
            return;
        };

        let buffer_width = out_w;
        let buffer_height = out_h;

        // Resize surface if needed (bind group managed internally by VM)
        if self.size != (buffer_width, buffer_height) {
            self.size = (buffer_width, buffer_height);
            gpu.surface.width = buffer_width;
            gpu.surface.height = buffer_height;
            gpu.surface.ensure_gpu_with(&gpu.device);
        }

        let (w, h) = self.size;

        // Delegate rendering to all VM layers in order (each overlays the previous result)
        SceneVM::draw_all_vms(
            base_vm,
            overlays,
            &gpu.device,
            &gpu.queue,
            &mut gpu.surface,
            w,
            h,
        );

        // Readback into the surface's CPU memory (blocking on native, non-blocking noop on wasm)
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        gpu.surface.download_from_gpu_with(&device, &queue);

        // On native, pixels are now in `surface.data`; copy them to the output buffer.
        // On WASM, if you need the pixels immediately, prefer `draw_async`.
        gpu.surface.copy_to_slice(out_pixels, out_w, out_h);
    }

    /// Cross-platform async render: same call on native & WASM.
    #[cfg(target_arch = "wasm32")]
    pub async fn render_frame_async(&mut self, out_pixels: &mut [u8], out_w: u32, out_h: u32) {
        let (gpu_slot, base_vm, overlays) = (&mut self.gpu, &mut self.vm, &mut self.overlay_vms);
        let Some(gpu) = gpu_slot.as_mut() else {
            return;
        };
        let buffer_width = out_w;
        let buffer_height = out_h;

        if self.size != (buffer_width, buffer_height) {
            self.size = (buffer_width, buffer_height);
            gpu.surface.width = buffer_width;
            gpu.surface.height = buffer_height;
            gpu.surface.ensure_gpu_with(&gpu.device);
        }

        let (w, h) = self.size;
        SceneVM::draw_all_vms(
            base_vm,
            overlays,
            &gpu.device,
            &gpu.queue,
            &mut gpu.surface,
            w,
            h,
        );

        // Start readback and await readiness
        let device = gpu.device.clone();
        let queue = gpu.queue.clone();
        gpu.surface.download_from_gpu_with(&device, &queue);
        let flag = gpu
            .surface
            .gpu
            .as_ref()
            .and_then(|g| g.map_ready.as_ref().map(|f| std::rc::Rc::clone(f)));
        if let Some(flag) = flag {
            MapReadyFuture { flag }.await;
        }
        let _ = gpu.surface.try_finish_download_from_gpu();
        gpu.surface.copy_to_slice(out_pixels, out_w, out_h);
    }

    /// Single cross-platform async entrypoint for rendering a frame.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn render_frame_async(&mut self, out_pixels: &mut [u8], out_w: u32, out_h: u32) {
        self.draw(out_pixels, out_w, out_h);
    }

    /// Cross-platform synchronous render entrypoint (one function for Native & WASM). Returns a RenderResult.
    /// Native: blocks until pixels are ready. WASM: presents the last completed frame
    /// and kicks off a new GPU frame if none is in flight. Call this every frame.
    /// On WASM, you must call `init_async().await` once before rendering.
    pub fn render_frame(&mut self, out_pixels: &mut [u8], out_w: u32, out_h: u32) -> RenderResult {
        #[cfg(not(target_arch = "wasm32"))]
        {
            // Native path just does the full render and readback synchronously
            self.draw(out_pixels, out_w, out_h);
            return RenderResult::Presented;
        }

        #[cfg(target_arch = "wasm32")]
        {
            // WASM path: auto-init GPU if needed, else non-blocking render logic.
            if self.gpu.is_none() {
                if !self.init_in_flight && self.needs_gpu_init {
                    self.init_in_flight = true;
                    let this: *mut SceneVM = self as *mut _;
                    spawn_local(async move {
                        // SAFETY: we rely on the caller to call `render_frame` from the UI thread.
                        // We only flip flags and build GPU state; no aliasing mutable accesses occur concurrently
                        // because the user code keeps calling `render_frame`, which is single-threaded on wasm.
                        unsafe {
                            (&mut *this).init_async().await;
                            (&mut *this).init_in_flight = false;
                        }
                    });
                }
                // Nothing to render until init finishes; return quietly.
                return RenderResult::InitPending;
            }
            let (gpu_slot, base_vm, overlays) =
                (&mut self.gpu, &mut self.vm, &mut self.overlay_vms);
            let gpu = gpu_slot.as_mut().unwrap();

            // Ensure surface size (bind group managed internally by VM)
            if self.size != (out_w, out_h) {
                self.size = (out_w, out_h);
                gpu.surface.width = out_w;
                gpu.surface.height = out_h;
                gpu.surface.ensure_gpu_with(&gpu.device);
            }

            // If a previous async map completed, finalize the download now.
            let ready = gpu
                .surface
                .gpu
                .as_ref()
                .and_then(|g| g.map_ready.as_ref())
                .map(|f| f.get())
                .unwrap_or(false);
            if ready {
                let _ = gpu.surface.try_finish_download_from_gpu();
            }

            // Determine the result for this frame without re-borrowing `self`
            let inflight_now = gpu
                .surface
                .gpu
                .as_ref()
                .and_then(|g| g.map_ready.as_ref())
                .is_some();
            let mut result = if ready {
                RenderResult::Presented
            } else if inflight_now {
                RenderResult::ReadbackPending
            } else {
                RenderResult::Presented
            };

            // Present whatever CPU pixels we currently have.
            gpu.surface.copy_to_slice(out_pixels, out_w, out_h);

            // If no readback is currently in flight, kick off one for the next frame.
            let inflight = gpu
                .surface
                .gpu
                .as_ref()
                .and_then(|g| g.map_ready.as_ref())
                .is_some();
            if !inflight {
                // Delegate rendering to the VM (compute 2D/3D chosen by VM::render_mode)
                let (w, h) = self.size;
                SceneVM::draw_all_vms(
                    base_vm,
                    overlays,
                    &gpu.device,
                    &gpu.queue,
                    &mut gpu.surface,
                    w,
                    h,
                );

                // Start non-blocking readback into the surface texture (map_async sets the flag)
                let device = gpu.device.clone();
                let queue = gpu.queue.clone();
                gpu.surface.download_from_gpu_with(&device, &queue);
                result = RenderResult::ReadbackPending;
            }
            return result;
        }
    }

    /// Load an image from various inputs (file path on native, raw bytes, &str) and decode to RGBA8.
    pub fn load_image_rgba<I: IntoDataInput>(&self, input: I) -> Option<(Vec<u8>, u32, u32)> {
        let bytes = match input.load_data() {
            Ok(b) => b,
            Err(_) => return None,
        };
        let img = match image::load_from_memory(&bytes) {
            Ok(i) => i,
            Err(_) => return None,
        };
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        Some((rgba.into_raw(), w, h))
    }
}

// --- Global GPU helpers ---
#[cfg(target_arch = "wasm32")]
fn global_gpu_get() -> Option<GlobalGpu> {
    GLOBAL_GPU_WASM.with(|c| c.borrow().clone())
}

#[cfg(target_arch = "wasm32")]
async fn global_gpu_init_async() {
    if global_gpu_get().is_some() {
        return;
    }
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("No compatible GPU adapter found (WebGPU)");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("scenevm-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("Failed to create wgpu device (WebGPU)");
    let gg = GlobalGpu {
        instance,
        adapter,
        device,
        queue,
    };
    GLOBAL_GPU_WASM.with(|c| *c.borrow_mut() = Some(gg));
}
impl SceneVM {
    fn for_each_vm_mut(&mut self, mut f: impl FnMut(&mut VM)) {
        f(&mut self.vm);
        for vm in &mut self.overlay_vms {
            f(vm);
        }
    }

    fn atom_touches_atlas(atom: &Atom) -> bool {
        matches!(
            atom,
            Atom::AddTile { .. }
                | Atom::AddSolid { .. }
                | Atom::SetTileMaterialFrames { .. }
                | Atom::BuildAtlas
                | Atom::Clear
                | Atom::ClearTiles
        )
    }
}
