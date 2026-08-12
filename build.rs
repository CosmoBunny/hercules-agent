// build.rs — builds llama.cpp as static libraries and links them into the
// Hercules binary, so no shared libllama.so / libloading dance is needed at
// runtime.
//
// Build flow:
//   1. Locate source: $LLAMA_CPP_SRC env var  →  ./llama.cpp submodule  →
//      clone from GitHub into OUT_DIR/llama.cpp-src.
//   2. Run CMake (Release, static libs, CPU-only safe defaults).
//   3. Build only the library targets we need (llama + ggml stack) — NOT the
//      llama-cli / llama-server executables that newer llama.cpp builds by
//      default and that drag in missing -lllama-server-impl / -lllama-cli-impl.
//   4. Emit cargo:rustc-link-* directives for every .a produced.
//
// The static build is gated behind the feature flag `llama-cpp-static`.
// Without the flag the old runtime dlopen path (`llama-cpp-bindings`) is used
// and this script is a no-op for that path.
//
// Environment variables:
//   LLAMA_CPP_SRC   – path to an existing llama.cpp checkout with CMakeLists.txt
//   LLAMA_CUDA=1    – enable CUDA backend
//   LLAMA_VULKAN=1  – enable Vulkan backend

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_SRC");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_VULKAN");

    if std::env::var("CARGO_FEATURE_LLAMA_CPP_STATIC").is_err() {
        return; // runtime dlopen path; nothing to do
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let src_dir = locate_or_fetch_source(&out_dir);
    let build_dir = out_dir.join("llama-cpp-build");

    cmake_configure(&src_dir, &build_dir);
    cmake_build_libs(&build_dir);
    emit_link_directives(&build_dir);
}

// ---------------------------------------------------------------------------
// Source location
// ---------------------------------------------------------------------------

fn locate_or_fetch_source(out_dir: &Path) -> PathBuf {
    // 1. Explicit override — user points at a pre-cloned tree.
    if let Ok(p) = std::env::var("LLAMA_CPP_SRC") {
        let path = PathBuf::from(&p);
        if !path.exists() {
            panic!(
                "LLAMA_CPP_SRC={p} does not exist.\n\
                 Make sure the path is correct, or unset LLAMA_CPP_SRC to let\n\
                 build.rs clone llama.cpp automatically."
            );
        }
        if !path.join("CMakeLists.txt").exists() {
            panic!(
                "LLAMA_CPP_SRC={p} exists but has no CMakeLists.txt.\n\
                 Is this a proper llama.cpp source tree?"
            );
        }
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
        eprintln!("[build.rs] llama.cpp source not found locally — cloning from GitHub …");
        eprintln!("[build.rs] Tip: run  git submodule add https://github.com/ggerganov/llama.cpp.git");
        eprintln!("[build.rs] or set LLAMA_CPP_SRC=/path/to/llama.cpp to skip the download.");
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

    let cuda   = std::env::var("LLAMA_CUDA").unwrap_or_default();
    let vulkan = std::env::var("LLAMA_VULKAN").unwrap_or_default();

    // Detect whether Ninja is available; fall back to the platform default.
    let generator = if which("ninja") { "Ninja" } else { "Unix Makefiles" };

    let mut cmd = Command::new("cmake");
    cmd.current_dir(build)
        .arg(src)
        .arg(format!("-G{generator}"))
        .arg("-DCMAKE_BUILD_TYPE=Release")
        // ── Static libraries ──────────────────────────────────────────────
        // Both variable names are used across llama.cpp versions.
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DGGML_SHARED_LIBS=OFF")          // ggml CMake variable (newer)
        // ── Disable ALL application targets ──────────────────────────────
        // Newer llama.cpp (post b4000) has a top-level apps directory that
        // builds llama-cli / llama-server and pulls in impl libs that only
        // exist when linking those executables.  Disable everything.
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .arg("-DLLAMA_BUILD_SERVER=OFF")
        .arg("-DLLAMA_BUILD_TOOLS=OFF")         // post-b4500 variable
        .arg("-DLLAMA_STANDALONE=OFF")
        // ── PIC required when linking into a Rust binary ──────────────────
        .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON");

    if cuda == "1" || cuda.eq_ignore_ascii_case("on") {
        cmd.arg("-DGGML_CUDA=ON");
    }
    if vulkan == "1" || vulkan.eq_ignore_ascii_case("on") {
        cmd.arg("-DGGML_VULKAN=ON");
    }

    let status = cmd.status().expect("cmake configure failed — is cmake installed?");
    assert!(status.success(), "CMake configure step failed");
}

// ---------------------------------------------------------------------------
// CMake build — library targets only
// ---------------------------------------------------------------------------

fn cmake_build_libs(build: &Path) {
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());

    // We build only the library targets we actually need, not the default `all`
    // target that would drag in llama-cli / llama-server executables.
    //
    // Target names are stable across llama.cpp versions:
    //   - `llama`    → src/libllama.a
    //   - `ggml`     → ggml/src/libggml.a (meta-lib pulling in ggml-base, ggml-cpu)
    //
    // Building `llama` is sufficient: CMake's dependency graph automatically
    // builds ggml, ggml-base, ggml-cpu etc. as transitive dependencies.
    let targets = ["llama", "ggml"];

    for target in &targets {
        let status = Command::new("cmake")
            .args([
                "--build", ".",
                "--config", "Release",
                "--target", target,
                "--parallel", &jobs,
            ])
            .current_dir(build)
            .status()
            .unwrap_or_else(|e| panic!("cmake --build --target {target} failed to spawn: {e}"));

        // `ggml` as a standalone target may not exist in all versions (it might
        // be a dependency-only interface).  Skip it gracefully; `llama` already
        // pulled it in.
        if !status.success() && *target == "ggml" {
            eprintln!("[build.rs] Note: cmake --target ggml failed (may be interface-only) — skipping");
        } else {
            assert!(
                status.success(),
                "cmake --build --target {target} failed.\n\
                 Tip: check the CMake output above for compiler or dependency errors."
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Link directives
// ---------------------------------------------------------------------------

/// Walk the build tree and emit `cargo:rustc-link-lib=static=<name>` for
/// every `.a` found (excluding CMake test/check archives), plus search paths.
fn emit_link_directives(build: &Path) {
    let archives = find_archives(build);
    assert!(
        !archives.is_empty(),
        "No .a files found under {} — CMake build may have produced nothing.\n\
         Try running the build manually:\n\
           cd {} && cmake --build . --target llama",
        build.display(),
        build.display(),
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

    // Link order: llama first (references ggml symbols), then ggml-*, then ggml.
    let mut sorted = archives.clone();
    sorted.sort_by_key(|p| {
        let s = p.file_stem().unwrap_or_default().to_string_lossy().to_lowercase();
        let s = s.strip_prefix("lib").unwrap_or(&s).to_string();
        if s == "llama"           { 0u8 }
        else if s.starts_with("ggml-") { 2 }
        else if s == "ggml"       { 3 }
        else                      { 1 }
    });

    for archive in &sorted {
        let stem = archive.file_stem().unwrap_or_default().to_string_lossy();
        let name = stem.strip_prefix("lib").unwrap_or(&stem);
        println!("cargo:rustc-link-lib=static={name}");
    }

    link_system_libs();
}

/// Recursively collect .a files, skipping CMake internal / test dirs.
fn find_archives(dir: &Path) -> Vec<PathBuf> {
    let mut results = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return results };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Skip CMake's own scratch directories.
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name == "CMakeFiles" || name == "_deps" || name == "Testing" {
                continue;
            }
            results.extend(find_archives(&path));
        } else if path.extension().map(|e| e == "a").unwrap_or(false) {
            results.push(path);
        }
    }
    results
}

fn link_system_libs() {
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");
    #[cfg(not(target_os = "macos"))]
    println!("cargo:rustc-link-lib=stdc++");
    #[cfg(unix)]
    println!("cargo:rustc-link-lib=pthread");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=m");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=dl");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn which(bin: &str) -> bool {
    Command::new("which").arg(bin).output().map(|o| o.status.success()).unwrap_or(false)
}
