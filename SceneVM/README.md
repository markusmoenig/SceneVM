# SceneVM Xcode Integration (SwiftUI, macOS / iOS / iPadOS)

This template drives SceneVM entirely on the GPU by presenting into a `CAMetalLayer` via the FFI exposed in `scenevm`.

## FFI Functions (from Rust)
```c
SceneVM* scenevm_ca_create(void* layer_ptr, uint32_t width, uint32_t height);
void     scenevm_ca_destroy(SceneVM* vm);
void     scenevm_ca_resize(SceneVM* vm, uint32_t width, uint32_t height);
int32_t  scenevm_ca_render(SceneVM* vm); // 0=presented, 1=init pending, 2=in-flight readback, -1=error
```

## Build the Rust static libraries
Run from repo root:
```bash
# macOS (Apple Silicon)
cargo build --release --target aarch64-apple-darwin

# macOS (Intel, if needed)
cargo build --release --target x86_64-apple-darwin

# iOS / iPadOS (device)
cargo build --release --target aarch64-apple-ios

# iOS / iPadOS (simulator; skip if you only need Apple Silicon and use arm64 simulator)
cargo build --release --target x86_64-apple-ios
```

Artifacts to add in Xcode → Build Phases → Link Binary With Libraries:
- `target/<triple>/release/libscenevm.a` for each platform/arch you target.

## Xcode project wiring (already in template)
- Swift files:
  - `SceneVM/SceneVM/SceneVMFFI.swift` (FFI declarations + wrapper)
  - `SceneVM/SceneVM/SceneVMView.swift` (SwiftUI host view with CAMetalLayer; uses CVDisplayLink/CADisplayLink)
- `SceneVM/SceneVM/ContentView.swift` (displays `SceneVMView`)
- `SceneVM/SceneVM/SceneVMApp.swift` (App entry)
- No bridging header required; FFI uses `@_silgen_name`.

## Linker / frameworks
- Add system frameworks: `Metal`, `QuartzCore` (plus AppKit/UIKit as usual).
- Ensure library search paths include the directories where the `.a` files live (e.g., `$(PROJECT_DIR)/../target/aarch64-apple-darwin/release` etc.).

## Demo app sharing (desktop / wasm / Xcode)
- The sample SceneVM app lives in `src/app.rs` and is re-exported as `scenevm::DemoApp`.
- `window-demo` (native + WASM) uses the same `DemoApp` through `run_scenevm_app`.
- The SwiftUI template calls the FFI runner which wraps the same `DemoApp`, so all three targets show identical content.

Runner symbols (used by `SceneVMFFI.swift`):
```c
void*  scenevm_runner_create(void* layer_ptr, uint32_t width, uint32_t height);
void   scenevm_runner_destroy(void* runner);
void   scenevm_runner_resize(void* runner, uint32_t width, uint32_t height);
int32_t scenevm_runner_render(void* runner); // 0=presented, 1=init pending, 2=in-flight, -1=error
```
To plug in your own app, swap `DemoApp` in `src/lib.rs` for the runner or add your own FFI entry points that build your app type.

## CAMetalLayer requirements
- `framebufferOnly = false` (already set in `SceneVMView.swift`).
- Pass drawable size = bounds * scale; `SceneVMView` handles this.

## Running
- macOS: build/run normally (links macOS static lib).
- iOS/iPadOS: select the appropriate scheme/device/simulator and ensure the matching static lib slice is linked.

## Notes
- The layer pointer passed to `scenevm_ca_create` must outlive the SceneVM instance.
- Rendering stays GPU-only; no CPU readback path is used here.
