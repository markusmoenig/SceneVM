pub mod bbox2d;
pub mod chunk;
pub mod intodata;
pub mod light;
pub mod material;
pub mod texture;
pub mod vm;

pub use crate::{
    bbox2d::BBox2D,
    chunk::Chunk,
    intodata::IntoDataInput,
    light::{Light, LightType},
    material::Material,
    texture::Texture,
    vm::{Atom, GeoId, Poly2D, Poly3D, RenderMode, VM},
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

    vm: VM,
}

impl Default for SceneVM {
    fn default() -> Self {
        Self::new(100, 100)
    }
}

impl SceneVM {
    /// Executes a single atom
    pub fn execute(&mut self, atom: Atom) {
        self.vm.execute(atom);
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
            Self {
                size: (initial_width, initial_height),
                gpu: None,
                needs_gpu_init: true,
                init_in_flight: false,
                vm: VM::new(4096, 4096),
            }
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

            Self {
                size: (initial_width, initial_height),
                gpu: Some(gpu),
                vm: VM::new(4096, 4096),
            }
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
        let Some(gpu) = self.gpu.as_mut() else {
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

        // Delegate rendering to the VM (compute 2D/3D chosen by VM::render_mode)
        self.vm
            .draw_into(&gpu.device, &gpu.queue, &mut gpu.surface, w, h);

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
        let Some(gpu) = self.gpu.as_mut() else {
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
        self.vm
            .draw_into(&gpu.device, &gpu.queue, &mut gpu.surface, w, h);

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
            let gpu = self.gpu.as_mut().unwrap();

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
                self.vm
                    .draw_into(&gpu.device, &gpu.queue, &mut gpu.surface, w, h);

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
