//
//  SceneVMApp.swift
//  SceneVM Unified Template
//
//  Created by Markus Moenig on 10/12/25.
//

import SwiftUI

@main
struct SceneVMApp: App {
    var body: some Scene {
        #if os(macOS)
        // macOS: Document-based app with NSDocument
        DocumentGroup(newDocument: { SceneVMDocument() }) { file in
            DocumentView(document: file.document)
        }
        #else
        // iOS: Document browser with UIDocument
        DocumentGroup(newDocument: SceneVMDocument()) { file in
            DocumentView(document: file.document)
                .ignoresSafeArea()
        }
        #endif
    }
}
