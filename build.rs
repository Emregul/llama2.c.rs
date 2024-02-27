use std::path::{Path, PathBuf};

fn main() {
    let mut build = cc::Build::new();
    let submodule_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("llama2-c");

    build.file(submodule_dir.join("run.c"));
    // build.file(submodule_dir.join("runq.c"));
    build.define("TESTING", None);
    build.compile("libllama");

    println!("cargo:rustc-link-lib=static=libllama");
    println!(
        "cargo:rustc-link-search=native={}",
        std::env::var("OUT_DIR").unwrap()
    );

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .allowlist_type("ssize_t")
        .allowlist_type("Config")
        .allowlist_type("TransformerWeights")
        .allowlist_type("RunState")
        .allowlist_type("Transformer")
        .allowlist_type("Tokenizer")
        .allowlist_type("ProbIndex")
        .allowlist_type("Sampler")
        .allowlist_function(".*")
        .size_t_is_usize(true)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
