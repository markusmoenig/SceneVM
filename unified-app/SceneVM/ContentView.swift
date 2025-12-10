//
//  ContentView.swift
//  SceneVM
//
//  Created by Markus Moenig on 10/12/25.
//

import SwiftUI
struct ContentView: View {
    var body: some View {
        #if os(macOS)
        SceneVMView() // respect the title bar inset for parity with winit
        #else
        SceneVMView()
            .ignoresSafeArea()
        #endif
    }
}

#Preview {
    ContentView()
}
