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
}
#endif
