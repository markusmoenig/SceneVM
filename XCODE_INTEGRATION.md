# Xcode Integration Guide

This guide explains how to integrate SceneVM with Xcode's document-based architecture for macOS and iOS/iPadOS.

## Architecture Overview

### Rust Side (SceneVM)
- **FFI Layer**: C functions exposed via `unified-app`
- **Data Flow**: JSON strings passed between Rust ↔ Swift
- **No File I/O in Rust**: All file operations handled by Swift

### Swift Side (Xcode)
- **Document Model**: NSDocument (macOS) / UIDocument (iOS)
- **File Management**: Automatic save, iCloud sync, versioning
- **Recent Files**: System-managed recent documents
- **Thumbnails**: Quick Look integration

## Platform Differences

| Feature | macOS | iOS/iPadOS |
|---------|-------|------------|
| Document Class | `NSDocument` | `UIDocument` |
| Multiple Windows | ✅ Yes | ✅ Yes (iPad) |
| Recent Files | NSDocumentController | System managed |
| File Dialogs | Native dialogs | Document picker |
| iCloud | CloudKit + NSDocument | UIDocument auto |
| Thumbnails | Quick Look | Quick Look |

## Implementation

### 1. Document Type Definition (Info.plist)

```xml
<key>CFBundleDocumentTypes</key>
<array>
    <dict>
        <key>CFBundleTypeName</key>
        <string>SceneVM Project</string>
        <key>CFBundleTypeRole</key>
        <string>Editor</string>
        <key>LSHandlerRank</key>
        <string>Owner</string>
        <key>LSItemContentTypes</key>
        <array>
            <string>com.yourcompany.scenevm-project</string>
        </array>
    </dict>
</array>

<key>UTExportedTypeDeclarations</key>
<array>
    <dict>
        <key>UTTypeConformsTo</key>
        <array>
            <string>public.json</string>
            <string>public.data</string>
        </array>
        <key>UTTypeDescription</key>
        <string>SceneVM Project</string>
        <key>UTTypeIdentifier</key>
        <string>com.yourcompany.scenevm-project</string>
        <key>UTTypeTagSpecification</key>
        <dict>
            <key>public.filename-extension</key>
            <array>
                <string>scenevm</string>
            </array>
        </dict>
    </dict>
</array>
```

### 2. macOS Document Class

```swift
// SceneVMDocument.swift (macOS)
import Cocoa

class SceneVMDocument: NSDocument {
    var sceneVMHandle: SceneVMHandle?
    private var metalLayer: CAMetalLayer?
    
    override init() {
        super.init()
        // Initial setup will happen in makeWindowControllers
    }
    
    override class var autosavesInPlace: Bool {
        return true
    }
    
    override func makeWindowControllers() {
        let storyboard = NSStoryboard(name: "Main", bundle: nil)
        if let windowController = storyboard.instantiateController(
            withIdentifier: "Document Window Controller"
        ) as? NSWindowController {
            addWindowController(windowController)
            
            // Setup SceneVM after window is created
            if let contentViewController = windowController.contentViewController as? SceneVMViewController {
                self.metalLayer = contentViewController.metalLayer
                setupSceneVM()
            }
        }
    }
    
    private func setupSceneVM() {
        guard let layer = metalLayer else { return }
        
        let size = layer.bounds.size
        let scale = layer.contentsScale
        
        sceneVMHandle = SceneVMHandle(
            layer: layer,
            size: size,
            scale: scale
        )
    }
    
    // MARK: - Reading
    
    override func read(from data: Data, ofType typeName: String) throws {
        guard let jsonString = String(data: data, encoding: .utf8) else {
            throw NSError(domain: "SceneVMDocument", code: 1, 
                         userInfo: [NSLocalizedDescriptionKey: "Invalid UTF-8 data"])
        }
        
        // Store JSON for loading after window setup
        pendingJSON = jsonString
    }
    
    private var pendingJSON: String?
    
    override func windowControllerDidLoadNib(_ windowController: NSWindowController) {
        super.windowControllerDidLoadNib(windowController)
        
        // Now we can load the project into SceneVM
        if let json = pendingJSON {
            let success = sceneVMHandle?.loadProject(json: json) ?? false
            if !success {
                print("Failed to load project")
            }
            pendingJSON = nil
        }
    }
    
    // MARK: - Writing
    
    override func data(ofType typeName: String) throws -> Data {
        guard let json = sceneVMHandle?.saveProject() else {
            throw NSError(domain: "SceneVMDocument", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to save project"])
        }
        
        guard let data = json.data(using: .utf8) else {
            throw NSError(domain: "SceneVMDocument", code: 3,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to encode JSON"])
        }
        
        return data
    }
    
    // MARK: - Thumbnails
    
    override func fileAttributesToWrite(
        to url: URL,
        ofType typeName: String,
        for saveOperation: NSDocument.SaveOperationType,
        originalContentsURL absoluteOriginalContentsURL: URL?
    ) throws -> [String : Any] {
        var attributes = try super.fileAttributesToWrite(
            to: url,
            ofType: typeName,
            for: saveOperation,
            originalContentsURL: absoluteOriginalContentsURL
        )
        
        // Generate thumbnail for Quick Look
        if let thumbnail = generateThumbnail() {
            attributes[NSFileAttributeKey.thumbnailDictionary.rawValue] = thumbnail
        }
        
        return attributes
    }
    
    private func generateThumbnail() -> [String: Any]? {
        // Request thumbnail from Rust
        // This would need a new FFI function
        return nil
    }
}
```

### 3. iOS/iPadOS Document Class

```swift
// SceneVMDocument.swift (iOS)
import UIKit

class SceneVMDocument: UIDocument {
    var jsonContent: String = "{}"
    var sceneVMHandle: SceneVMHandle?
    
    override func contents(forType typeName: String) throws -> Any {
        // Save current state to JSON
        if let json = sceneVMHandle?.saveProject() {
            jsonContent = json
        }
        
        guard let data = jsonContent.data(using: .utf8) else {
            throw NSError(domain: "SceneVMDocument", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to encode JSON"])
        }
        
        return data
    }
    
    override func load(fromContents contents: Any, ofType typeName: String?) throws {
        guard let data = contents as? Data,
              let json = String(data: data, encoding: .utf8) else {
            throw NSError(domain: "SceneVMDocument", code: 2,
                         userInfo: [NSLocalizedDescriptionKey: "Invalid document data"])
        }
        
        jsonContent = json
        
        // Apply to SceneVM if handle exists
        if let handle = sceneVMHandle {
            let success = handle.loadProject(json: json)
            if !success {
                print("Failed to load project into SceneVM")
            }
        }
    }
    
    // MARK: - Thumbnails
    
    override func fileAttributesToWrite(
        to url: URL,
        for saveOperation: UIDocument.SaveOperation
    ) throws -> [AnyHashable : Any] {
        var attributes = try super.fileAttributesToWrite(to: url, for: saveOperation)
        
        // Add thumbnail for iOS document browser
        if let thumbnail = generateThumbnail() {
            attributes[URLResourceKey.thumbnailDictionaryKey] = thumbnail
        }
        
        return attributes
    }
    
    private func generateThumbnail() -> [String: Any]? {
        // Generate thumbnail image
        return nil
    }
}
```

### 4. Document Browser (iOS/iPadOS)

```swift
// DocumentBrowserViewController.swift (iOS)
import UIKit

class DocumentBrowserViewController: UIDocumentBrowserViewController, 
                                     UIDocumentBrowserViewControllerDelegate {
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        delegate = self
        
        allowsDocumentCreation = true
        allowsPickingMultipleItems = false
        
        // Enable iCloud
        browserUserInterfaceStyle = .dark
    }
    
    // MARK: - UIDocumentBrowserViewControllerDelegate
    
    func documentBrowser(
        _ controller: UIDocumentBrowserViewController,
        didRequestDocumentCreationWithHandler importHandler: @escaping (URL?, UIDocumentBrowserViewController.ImportMode) -> Void
    ) {
        // Create new document
        let newDocumentURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("Untitled.scenevm")
        
        // Create empty project
        let emptyProject = "{}"
        try? emptyProject.write(to: newDocumentURL, atomically: true, encoding: .utf8)
        
        importHandler(newDocumentURL, .move)
    }
    
    func documentBrowser(
        _ controller: UIDocumentBrowserViewController,
        didPickDocumentsAt documentURLs: [URL]
    ) {
        guard let sourceURL = documentURLs.first else { return }
        
        // Open document
        presentDocument(at: sourceURL)
    }
    
    func documentBrowser(
        _ controller: UIDocumentBrowserViewController,
        didImportDocumentAt sourceURL: URL,
        toDestinationURL destinationURL: URL
    ) {
        presentDocument(at: destinationURL)
    }
    
    func documentBrowser(
        _ controller: UIDocumentBrowserViewController,
        failedToImportDocumentAt documentURL: URL,
        error: Error?
    ) {
        // Handle error
    }
    
    // MARK: - Document Presentation
    
    func presentDocument(at documentURL: URL) {
        let document = SceneVMDocument(fileURL: documentURL)
        
        document.open { success in
            if success {
                // Show document view controller
                let storyboard = UIStoryboard(name: "Main", bundle: nil)
                if let documentVC = storyboard.instantiateViewController(
                    withIdentifier: "DocumentViewController"
                ) as? DocumentViewController {
                    documentVC.document = document
                    
                    let navController = UINavigationController(rootViewController: documentVC)
                    navController.modalPresentationStyle = .fullScreen
                    
                    self.present(navController, animated: true)
                }
            } else {
                // Handle error
            }
        }
    }
}
```

### 5. View Controller Integration

```swift
// DocumentViewController.swift (iOS)
import UIKit
import MetalKit

class DocumentViewController: UIViewController {
    var document: SceneVMDocument?
    private var metalView: MTKView!
    private var displayLink: CADisplayLink?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupMetal()
        setupSceneVM()
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        document?.open { [weak self] success in
            if success {
                self?.loadDocumentIntoSceneVM()
            }
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        document?.close(completionHandler: nil)
    }
    
    private func setupMetal() {
        metalView = MTKView(frame: view.bounds, device: MTLCreateSystemDefaultDevice())
        metalView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(metalView)
    }
    
    private func setupSceneVM() {
        guard let metalLayer = metalView.layer as? CAMetalLayer else { return }
        
        let size = view.bounds.size
        let scale = view.window?.screen.scale ?? 1.0
        
        let handle = SceneVMHandle(layer: metalLayer, size: size, scale: scale)
        document?.sceneVMHandle = handle
        
        loadDocumentIntoSceneVM()
        startRenderLoop()
    }
    
    private func loadDocumentIntoSceneVM() {
        guard let document = document,
              let handle = document.sceneVMHandle else { return }
        
        let _ = handle.loadProject(json: document.jsonContent)
    }
    
    private func startRenderLoop() {
        displayLink = CADisplayLink(target: self, selector: #selector(renderFrame))
        displayLink?.add(to: .main, forMode: .common)
    }
    
    @objc private func renderFrame() {
        document?.sceneVMHandle?.render()
    }
    
    @IBAction func closeDocument(_ sender: Any) {
        dismiss(animated: true)
    }
}
```

## iCloud Integration

### Enable iCloud in Xcode
1. Select project → Signing & Capabilities
2. Add "iCloud" capability
3. Check "iCloud Documents"
4. Add container: `iCloud.com.yourcompany.SceneVM`

### Automatic iCloud Sync
```swift
// Documents automatically sync when using:
// - NSDocument with ubiquityContainerURL
// - UIDocument in iCloud container

// Check if iCloud is available
if FileManager.default.ubiquityIdentityToken != nil {
    print("iCloud available")
}
```

## Recent Files

### macOS
```swift
// Automatic via NSDocument
// Recent files appear in File → Open Recent

// To add manually:
NSDocumentController.shared.noteNewRecentDocumentURL(url)
```

### iOS
```swift
// System manages recent files in document browser
// No manual intervention needed
```

## Thumbnails

### Quick Look Preview (macOS & iOS)
```swift
import QuickLook

extension SceneVMDocument: QLPreviewingController {
    func preparePreviewOfFile(
        at url: URL,
        completionHandler handler: @escaping (Error?) -> Void
    ) {
        // Generate preview image
        if let thumbnail = generateThumbnailImage() {
            // Save thumbnail
            handler(nil)
        }
    }
    
    private func generateThumbnailImage() -> UIImage? {
        // Request from Rust FFI
        // Would need new FFI function:
        // unified_app_runner_generate_thumbnail()
        return nil
    }
}
```

## File Format

### .scenevm File Structure
```json
{
  "metadata": {
    "name": "My Project",
    "version": "1.0.0",
    "app_version": "0.1.0",
    "created_at": 1702345678,
    "modified_at": 1702345999,
    "author": "User Name",
    "description": "Project description"
  },
  "data": {
    "slider_value": 75.5,
    "settings": ["option1", "option2"]
  }
}
```

## Testing

### macOS
1. Run app
2. File → New (⌘N)
3. Edit project
4. File → Save (⌘S)
5. Close window
6. File → Open Recent

### iOS/iPadOS
1. Run app → Document browser opens
2. Tap "+" → Create new
3. Edit project
4. Tap "< Documents" → Auto-saves
5. Document appears in browser with thumbnail

## Best Practices

1. **Auto-save**: Enable for seamless experience
2. **iCloud**: Test with and without iCloud
3. **Thumbnails**: Generate 256x256 images
4. **Error Handling**: Handle corrupted files gracefully
5. **Migration**: Support older file versions
6. **Large Files**: Consider file packages for images/assets

## Troubleshooting

**Issue**: Documents don't open
- Check UTI declaration in Info.plist
- Verify file extension matches

**Issue**: iCloud not syncing
- Check iCloud entitlements
- Verify container ID matches

**Issue**: Recent files not showing
- Use `noteNewRecentDocumentURL` on macOS
- iOS handles automatically

**Issue**: Thumbnails not displaying
- Implement Quick Look extension
- Check thumbnail generation code
