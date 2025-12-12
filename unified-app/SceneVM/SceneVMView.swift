import SwiftUI
import QuartzCore
import Metal

struct SceneVMView: View {
    var body: some View {
        PlatformView()
    }
}

#if os(macOS)
struct PlatformView: NSViewRepresentable {
    func makeNSView(context: Context) -> MetalContainer {
        MetalContainer()
    }

    func updateNSView(_ nsView: MetalContainer, context: Context) {}
}

final class MetalContainer: NSView {
    private let metalLayer = CAMetalLayer()
    private var handle: SceneVMHandle?
    private var displayLink: CVDisplayLink?
    private var pinchScaleAccumulator: CGFloat = 1.0

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        metalLayer.device = MTLCreateSystemDefaultDevice()
        metalLayer.pixelFormat = .bgra8Unorm
        metalLayer.framebufferOnly = false
        layer = metalLayer

        CVDisplayLinkCreateWithActiveCGDisplays(&displayLink)
        CVDisplayLinkSetOutputHandler(displayLink!) { [weak self] _, _, _, _, _ in
            DispatchQueue.main.async { self?.drawFrame() }
            return kCVReturnSuccess
        }
        CVDisplayLinkStart(displayLink!)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layout() {
        super.layout()
        metalLayer.frame = bounds
        let scale = window?.backingScaleFactor ?? NSScreen.main?.backingScaleFactor ?? 2.0
        metalLayer.contentsScale = scale
        metalLayer.drawableSize = CGSize(width: bounds.width * scale, height: bounds.height * scale)
        if handle == nil && bounds.width > 0 && bounds.height > 0 {
            handle = SceneVMHandle(layer: metalLayer, size: bounds.size, scale: scale)
        } else {
            handle?.resize(to: bounds.size, scale: scale)
        }
    }

    private func drawFrame() {
        handle?.render()
    }

    private func toLogicalCoords(_ point: NSPoint) -> (CGFloat, CGFloat) {
        // NSView event locations are already in logical points; flip to top-left.
        let x = point.x
        let y = bounds.height - point.y
        return (x, y)
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach { removeTrackingArea($0) }
        let options: NSTrackingArea.Options = [.mouseMoved, .activeInKeyWindow, .inVisibleRect]
        let area = NSTrackingArea(rect: bounds, options: options, owner: self, userInfo: nil)
        addTrackingArea(area)
    }

    override func mouseMoved(with event: NSEvent) {
        let loc = convert(event.locationInWindow, from: nil)
        let (x, y) = toLogicalCoords(loc)
        handle?.mouseMove(x: x, y: y)
    }

    override func mouseDown(with event: NSEvent) {
        let loc = convert(event.locationInWindow, from: nil)
        let (x, y) = toLogicalCoords(loc)
        handle?.mouseDown(x: x, y: y)
    }

    override func mouseDragged(with event: NSEvent) {
        let loc = convert(event.locationInWindow, from: nil)
        let (x, y) = toLogicalCoords(loc)
        handle?.mouseMove(x: x, y: y)
    }

    override func mouseUp(with event: NSEvent) {
        let loc = convert(event.locationInWindow, from: nil)
        let (x, y) = toLogicalCoords(loc)
        handle?.mouseUp(x: x, y: y)
    }

    override func scrollWheel(with event: NSEvent) {
        handle?.scroll(dx: event.scrollingDeltaX, dy: event.scrollingDeltaY)
    }

    override func magnify(with event: NSEvent) {
        let loc = convert(event.locationInWindow, from: nil)
        let (x, y) = toLogicalCoords(loc)
        pinchScaleAccumulator += event.magnification
        let scale = max(0.01, 1.0 + pinchScaleAccumulator)
        handle?.pinch(scale: scale, center: CGPoint(x: x, y: y))
    }

    deinit {
        if let dl = displayLink {
            CVDisplayLinkStop(dl)
        }
    }
}
#else
struct PlatformView: UIViewRepresentable {
    func makeUIView(context: Context) -> MetalContainer {
        MetalContainer()
    }

    func updateUIView(_ uiView: MetalContainer, context: Context) {}
}

final class MetalContainer: UIView {
    private var metalLayer: CAMetalLayer { layer as! CAMetalLayer }
    private var handle: SceneVMHandle?
    private var displayLink: CADisplayLink?
    private var pinchRecognizer: UIPinchGestureRecognizer?

    override class var layerClass: AnyClass { CAMetalLayer.self }

    override init(frame: CGRect) {
        super.init(frame: frame)
        let layer = metalLayer
        layer.pixelFormat = .bgra8Unorm
        layer.framebufferOnly = false
        layer.device = MTLCreateSystemDefaultDevice()
        let scale = UIScreen.main.scale
        layer.contentsScale = scale

        displayLink = CADisplayLink(target: self, selector: #selector(tick))
        displayLink?.add(to: .main, forMode: .common)

        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
        addGestureRecognizer(pinch)
        pinchRecognizer = pinch
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        metalLayer.frame = bounds
        let scale = window?.screen.scale ?? UIScreen.main.scale
        metalLayer.contentsScale = scale
        metalLayer.drawableSize = CGSize(width: bounds.width * scale, height: bounds.height * scale)
        if handle == nil && bounds.width > 0 && bounds.height > 0 {
            handle = SceneVMHandle(layer: metalLayer, size: bounds.size, scale: scale)
        } else {
            handle?.resize(to: bounds.size, scale: scale)
        }
    }

    @objc private func tick() {
        handle?.render()
    }

    private func toLogicalCoords(_ point: CGPoint) -> (CGFloat, CGFloat) {
        // UIKit coordinates are already logical with origin top-left.
        return (point.x, point.y)
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = touches.first else { return }
        let loc = t.location(in: self)
        let (x, y) = toLogicalCoords(loc)
        handle?.mouseDown(x: x, y: y)
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = touches.first else { return }
        let loc = t.location(in: self)
        let (x, y) = toLogicalCoords(loc)
        handle?.mouseMove(x: x, y: y)
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = touches.first else { return }
        let loc = t.location(in: self)
        let (x, y) = toLogicalCoords(loc)
        handle?.mouseUp(x: x, y: y)
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let t = touches.first else { return }
        let loc = t.location(in: self)
        let (x, y) = toLogicalCoords(loc)
        handle?.mouseUp(x: x, y: y)
    }

    @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
        let loc = gesture.location(in: self)
        let (x, y) = toLogicalCoords(loc)
        handle?.pinch(scale: gesture.scale, center: CGPoint(x: x, y: y))
    }
}
#endif
