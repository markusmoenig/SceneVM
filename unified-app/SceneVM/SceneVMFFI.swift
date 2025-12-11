import Foundation
import QuartzCore

// C FFI imported directly via @_silgen_name to avoid a bridging header.
@_silgen_name("unified_app_runner_create")
func unified_app_runner_create(_ layer_ptr: UnsafeMutableRawPointer?, _ width: UInt32, _ height: UInt32) -> UnsafeMutableRawPointer?

@_silgen_name("unified_app_runner_destroy")
func unified_app_runner_destroy(_ vm: UnsafeMutableRawPointer?)

@_silgen_name("unified_app_runner_resize")
func unified_app_runner_resize(_ vm: UnsafeMutableRawPointer?, _ width: UInt32, _ height: UInt32)

@_silgen_name("unified_app_runner_render")
func unified_app_runner_render(_ vm: UnsafeMutableRawPointer?) -> Int32

@_silgen_name("unified_app_runner_mouse_down")
func unified_app_runner_mouse_down(_ vm: UnsafeMutableRawPointer?, _ x: Float, _ y: Float)

@_silgen_name("unified_app_runner_mouse_up")
func unified_app_runner_mouse_up(_ vm: UnsafeMutableRawPointer?, _ x: Float, _ y: Float)

@_silgen_name("unified_app_runner_mouse_move")
func unified_app_runner_mouse_move(_ vm: UnsafeMutableRawPointer?, _ x: Float, _ y: Float)

@_silgen_name("unified_app_runner_scroll")
func unified_app_runner_scroll(_ vm: UnsafeMutableRawPointer?, _ dx: Float, _ dy: Float)

@_silgen_name("unified_app_runner_pinch")
func unified_app_runner_pinch(_ vm: UnsafeMutableRawPointer?, _ scale: Float, _ center_x: Float, _ center_y: Float)

/// Thin Swift wrapper around the SceneVM FFI for CAMetalLayer presentation.
final class SceneVMHandle {
    private var vm: UnsafeMutableRawPointer?
    private weak var layer: CAMetalLayer?

    init?(layer: CAMetalLayer, size: CGSize, scale: CGFloat) {
        let ptr = Unmanaged.passUnretained(layer).toOpaque()
        let w = UInt32(max(max(layer.drawableSize.width, size.width * scale), 1))
        let h = UInt32(max(max(layer.drawableSize.height, size.height * scale), 1))
        guard let handle = unified_app_runner_create(ptr, w, h) else {
            return nil
        }
        self.layer = layer
        self.vm = handle
    }

    func resize(to size: CGSize, scale: CGFloat) {
        guard let vm else { return }
        let drawable = layer?.drawableSize ?? CGSize(width: size.width * scale, height: size.height * scale)
        let w = UInt32(max(drawable.width, 1))
        let h = UInt32(max(drawable.height, 1))
        unified_app_runner_resize(vm, w, h)
    }

    func render() {
        guard let vm else { return }
        _ = unified_app_runner_render(vm)
    }

    func mouseDown(x: CGFloat, y: CGFloat) {
        guard let vm else { return }
        unified_app_runner_mouse_down(vm, Float(x), Float(y))
    }

    func mouseUp(x: CGFloat, y: CGFloat) {
        guard let vm else { return }
        unified_app_runner_mouse_up(vm, Float(x), Float(y))
    }

    func mouseMove(x: CGFloat, y: CGFloat) {
        guard let vm else { return }
        unified_app_runner_mouse_move(vm, Float(x), Float(y))
    }

    func scroll(dx: CGFloat, dy: CGFloat) {
        guard let vm else { return }
        unified_app_runner_scroll(vm, Float(dx), Float(dy))
    }

    func pinch(scale: CGFloat, center: CGPoint) {
        guard let vm else { return }
        unified_app_runner_pinch(vm, Float(scale), Float(center.x), Float(center.y))
    }

    deinit {
        if let vm {
            unified_app_runner_destroy(vm)
        }
    }
}
