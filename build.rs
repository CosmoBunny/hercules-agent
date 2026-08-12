// build.rs — builds llama.cpp as static libraries and links them into the
// Hercules binary, so no shared libllama.so / libloading dance is needed at
// runtime.
//
// Build flow:
//   1. Locate source: $LLAMA_CPP_SRC env var  →  ./llama.cpp submodule  →
//      clone from GitHub into OUT_DIR/llama.cpp-src.
//   2. Run CMake (Release, static libs, CPU-only safe defaults).
//   3. Emit cargo:rustc-link-* directives for every .a produced.
//
// The static build is gated behind the feature flag `llama-cpp-static`.
// Without the flag the old runtime dlopen path (`llama-cpp-bindings`) is used
// and this script is a no-op for that path.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Re-run if build.rs itself changes or the relevant env var is set.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_SRC");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_VULKAN");

    // Only build statically when the feature is requested.
    if std::env::var("CARGO_FEATURE_LLAMA_CPP_STATIC").is_err() {
        // Feature not enabled — nothing to do (runtime dlopen path is active).
        return;
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let src_dir = locate_or_fetch_source(&out_dir);
    let build_dir = out_dir.join("llama-cpp-build");

    cmake_configure(&src_dir, &build_dir);
    cmake_build(&build_dir);
    emit_link_directives(&build_dir);
}

// ---------------------------------------------------------------------------
// Source location
// ---------------------------------------------------------------------------

fn locate_or_fetch_source(out_dir: &Path) -> PathBuf {
    // 1. Explicit override — user points at a pre-cloned tree.
    if let Ok(p) = std::env::var("LLAMA_CPP_SRC") {
        let path = PathBuf::from(&p);
        assert!(
            path.join("CMakeLists.txt").exists(),
            "LLAMA_CPP_SRC={p} does not contain CMakeLists.txt"
        );
        println!("cargo:rerun-if-changed={p}/CMakeLists.txt");
        return path;
    }

    // 2. Git submodule at ./llama.cpp (preferred for reproducible builds).
    let submodule = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("llama.cpp");
    if submodule.join("CMakeLists.txt").exists() {
        println!("cargo:rerun-if-changed=llama.cpp/CMakeLists.txt");
        return submodule;
    }

    // 3. Auto-clone into OUT_DIR (works in CI without pre-cloning).
    let clone_target = out_dir.join("llama.cpp-src");
    if !clone_target.join("CMakeLists.txt").exists() {
        eprintln!("[build.rs] Cloning llama.cpp …");
        let status = Command::new("git")
            .args([
                "clone",
                "--depth=1",
                "--branch", "master",
                "https://github.com/ggerganov/llama.cpp.git",
                clone_target.to_str().unwrap(),
            ])
            .status()
            .expect("Failed to run git clone — ensure git is in PATH");
        assert!(status.success(), "git clone llama.cpp failed");
    }
    clone_target
}

// ---------------------------------------------------------------------------
// CMake configure
// ---------------------------------------------------------------------------

fn cmake_configure(src: &Path, build: &Path) {
    std::fs::create_dir_all(build).expect("create build dir");

    // Optional GPU backends via environment.
    let cuda    = std::env::var("LLAMA_CUDA").unwrap_or_default();
    let vulkan  = std::env::var("LLAMA_VULKAN").unwrap_or_default();

    let mut cmd = Command::new("cmake");
    cmd.current_dir(build)
        .arg(src)
        // Build type
        .arg("-DCMAKE_BUILD_TYPE=Release")
        // Always produce static libraries.
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DLLAMA_BUILD_SHARED_LIBS=OFF")
        // Don't build llama.cpp's own examples / tests — saves minutes.
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .arg("-DLLAMA_BUILD_SERVER=OFF")
        // Use Ninja if available for faster parallel builds.
        .arg("-GNinja")
        // Position-independent code required when linking into a Rust binary.
        .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON");

    // GPU backends
    if cuda == "1" || cuda.eq_ignore_ascii_case("on") {
        cmd.arg("-DGGML_CUDA=ON");
    }
    if vulkan == "1" || vulkan.eq_ignore_ascii_case("on") {
        cmd.arg("-DGGML_VULKAN=ON");
    }

    let status = cmd.status().expect("cmake configure failed — ensure cmake is in PATH");
    assert!(status.success(), "CMake configure step failed");
}

// ---------------------------------------------------------------------------
// CMake build
// ---------------------------------------------------------------------------

fn cmake_build(build: &Path) {
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());

    let status = Command::new("cmake")
        .args(["--build", ".", "--config", "Release", "--parallel", &jobs])
        .current_dir(build)
        .status()
        .expect("cmake --build failed");
    assert!(status.success(), "CMake build step failed");
}

// ---------------------------------------------------------------------------
// Link directives
// ---------------------------------------------------------------------------

/// Walk the build tree and emit `cargo:rustc-link-lib=static=<name>` for
/// every `.a` found, plus the link-search path.
fn emit_link_directives(build: &Path) {
    // Collect all static archives produced by the build.
    let archives = find_archives(build);
    assert!(
        !archives.is_empty(),
        "No .a files found under {} — CMake build may have failed",
        build.display()
    );

    // Deduplicate library directories.
    let mut lib_dirs: Vec<PathBuf> = Vec::new();
    for archive in &archives {
        let dir = archive.parent().unwrap().to_path_buf();
        if !lib_dirs.contains(&dir) {
            lib_dirs.push(dir);
        }
    }

    for dir in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    // Link order matters: llama → ggml-* → ggml → system libs.
    // Sort so llama.a comes first, then ggml libs, then the rest.
    let mut sorted = archives.clone();
    sorted.sort_by_key(|p| {
        let stem = p.file_stem().unwrap_or_default().to_string_lossy().to_lowercase();
        // Lower number = linked earlier (higher priority).
        if stem == "llama"            { 0u8 }
        else if stem.starts_with("ggml-") { 2 }
        else if stem == "ggml"        { 3 }
        else                          { 1 }
    });

    for archive in &sorted {
        // Strip leading "lib" from the stem (libllama.a → llama).
        let stem = archive
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        let name = stem.strip_prefix("lib").unwrap_or(&stem);
        println!("cargo:rustc-link-lib=static={name}");
    }

    // System libraries llama.cpp needs.
    link_system_libs();
}

fn find_archives(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                results.extend(find_archives(&path));
            } else if path.extension().map(|e| e == "a").unwrap_or(false) {
                results.push(path);
            }
        }
    }
    results
}

fn link_system_libs() {
    // C++ standard library — required on all platforms.
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");
    #[cfg(not(target_os = "macos"))]
    println!("cargo:rustc-link-lib=stdc++");

    // POSIX threading.
    #[cfg(unix)]
    println!("cargo:rustc-link-lib=pthread");

    // Math library (needed by ggml on Linux).
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=m");

    // Dynamic loading (needed only for the dlopen path, not static — kept for
    // completeness in case ggml links it transitively).
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dl");
}
