use scenevm::{DemoApp, run_scenevm_app};

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    run_scenevm_app(DemoApp::new())?;
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn wasm_main() -> Result<(), wasm_bindgen::JsValue> {
    run_scenevm_app(DemoApp::new())
}

#[cfg(target_arch = "wasm32")]
fn main() {
    let _ = wasm_main();
}
