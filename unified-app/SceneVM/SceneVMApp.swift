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
        // Both macOS and iOS use the same closure-based syntax
        DocumentGroup(newDocument: { SceneVMDocument() }) { file in
            DocumentView(document: file.document)
                #if os(iOS)
                .ignoresSafeArea()
                #endif
        }
    }
}
