// build.rs — links llama.cpp into the Hercules binary at build time so the
// final executable is self-contained (no runtime library search required).
//
// Two modes, tried in order:
//
//   A) Pre-built install  (LLAMA_INSTALL_DIR)
//      Point at a directory that already contains .so/.dll / .a/.lib files.
//      If static archives are found → static link (truly self-contained).
//      If only shared libs found    → dynamic link + bake RPATH/path hint.
//      Examples:
//        Linux/macOS:
//          LLAMA_INSTALL_DIR=/opt/llama.cpp cargo build --features llama-cpp-static
//        Windows (PowerShell):
//          $env:LLAMA_INSTALL_DIR="C:\llama.cpp"
//          cargo build --features llama-cpp-static
//
//   B) Build from source  (LLAMA_CPP_SRC / submodule / auto-clone)
//      Requires cmake + a C++17 compiler.  Produces static archives.
//      Examples:
//        # submodule (recommended)
//        git submodule add https://github.com/ggerganov/llama.cpp.git
//        cargo build --release --features llama-cpp-static
//
//        # existing source checkout
//        LLAMA_CPP_SRC=~/src/llama.cpp cargo build --release --features llama-cpp-static
//
//        # auto-clone (needs internet on first build)
//        cargo build --release --features llama-cpp-static
//
// The feature flag `llama-cpp-static` must be enabled; without it the runtime
// dlopen/LoadLibrary path is used and this script does nothing.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_INSTALL_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_SRC");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_VULKAN");

    if std::env::var("CARGO_FEATURE_LLAMA_CPP_STATIC").is_err() {
        return; // runtime dlopen path active; nothing to do
    }

    // ── Mode A: pre-built install directory ──────────────────────────────────
    if let Ok(dir) = std::env::var("LLAMA_INSTALL_DIR") {
        let path = PathBuf::from(&dir);
        if !path.exists() {
            panic!(
                "\nLLAMA_INSTALL_DIR={dir} does not exist.\n\
                 Set it to the directory containing libllama.so / llama.lib / libllama.a.\n"
            );
        }
        link_from_install_dir(&path);
        return;
    }

    // ── Mode B: build from source ─────────────────────────────────────────────
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let src_dir = locate_or_fetch_source(&out_dir);
    let build_dir = out_dir.join("llama-cpp-build");

    cmake_configure(&src_dir, &build_dir);
    cmake_build_libs(&build_dir);
    emit_link_directives(&build_dir);
}

// ===========================================================================
// Mode A — pre-built install
// ===========================================================================

fn link_from_install_dir(dir: &Path) {
    // llama.cpp core libraries we care about (ignores openvino, tbb, hwloc…)
    let core_names = [
        "llama", "ggml", "ggml-base", "ggml-cpu",
        "ggml-rpc", "ggml-cuda", "ggml-vulkan", "ggml-metal",
        "llama-common",
    ];

    println!("cargo:rustc-link-search=native={}", dir.display());

    // Collect static archives: .a (Unix/MinGW) or .lib (MSVC)
    let mut archives = collect_libs(dir, "a");
    archives.extend(collect_libs(dir, "lib")); // MSVC static libs

    // Collect shared libs: .so (Linux) / .dylib (macOS) / .dll.lib or .dll (Windows)
    let shared = collect_libs(dir, shared_ext());

    if !archives.is_empty() {
        eprintln!(
            "[build.rs] {} static archive(s) in {} — static linking",
            archives.len(), dir.display()
        );
        for path in sort_libs(archives) {
            let stem = lib_stem(&path);
            if core_names.iter().any(|n| stem == *n) {
                println!("cargo:rustc-link-lib=static={stem}");
            }
        }
    } else if !shared.is_empty() {
        eprintln!(
            "[build.rs] No static archives in {} — dynamic linking with baked rpath",
            dir.display()
        );
        // Bake the directory so the binary finds the libs without PATH/LD_LIBRARY_PATH.
        bake_rpath(dir);

        for path in sort_libs(shared) {
            let stem = lib_stem(&path);
            if core_names.iter().any(|n| stem == *n) {
                println!("cargo:rustc-link-lib=dylib={stem}");
            }
        }
    } else {
        panic!(
            "\nLLAMA_INSTALL_DIR={} has no libllama.{{a,so,dylib,lib,dll}} files.\n\
             Is this the correct directory?\n",
            dir.display()
        );
    }

    link_system_libs();
}

/// Bake the library directory into the binary so it is found at runtime
/// without the user setting LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
fn bake_rpath(dir: &Path) {
    let d = dir.to_string_lossy();
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{d}");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{d}");
    // Windows uses PATH; there's no ELF rpath equivalent.
    // The DLLs must be alongside the .exe or in a directory on PATH.
    // We emit a warning so the packager knows.
    #[cfg(target_os = "windows")]
    eprintln!(
        "[build.rs] Windows: DLLs from {d} must be in the same directory as hercules.exe \
         or on the system PATH at runtime."
    );
}

fn collect_libs(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_file() { continue; }
        let ext_matches = p.extension().map(|e| e == ext).unwrap_or(false);
        if !ext_matches { continue; }
        let fname = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
        // Unix: name starts with "lib". Windows MSVC: just "llama.lib" etc.
        if fname.starts_with("lib") || fname.starts_with("llama") || fname.starts_with("ggml") {
            out.push(p);
        }
    }
    out
}

/// Platform-specific shared library extension.
fn shared_ext() -> &'static str {
    if cfg!(target_os = "windows") { "dll" }
    else if cfg!(target_os = "macos") { "dylib" }
    else { "so" }
}

/// Extract the logical library name from a path.
///   libllama.so.0.0.1 → llama
///   llama.lib          → llama
///   libggml-cpu.a      → ggml-cpu
fn lib_stem(p: &Path) -> String {
    let fname = p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
    // Strip "lib" prefix if present
    let s = fname.strip_prefix("lib").unwrap_or(&fname);
    // Take everything before the first '.'
    s.split('.').next().unwrap_or(s).to_string()
}

fn sort_libs(mut libs: Vec<PathBuf>) -> Vec<PathBuf> {
    libs.sort_by_key(|p| {
        let s = lib_stem(p);
        if s == "llama"               { 0u8 }
        else if s.starts_with("ggml-") { 2 }
        else if s == "ggml"            { 3 }
        else                           { 1 }
    });
    // Deduplicate by logical name (libllama.so / libllama.so.0 / libllama.so.0.0.1)
    let mut seen = HashSet::new();
    libs.retain(|p| seen.insert(lib_stem(p)));
    libs
}

// ===========================================================================
// Mode B — build from source
// ===========================================================================

fn locate_or_fetch_source(out_dir: &Path) -> PathBuf {
    // 1. Explicit source tree override.
    if let Ok(p) = std::env::var("LLAMA_CPP_SRC") {
        let path = PathBuf::from(&p);
        if !path.exists() {
            panic!(
                "\nLLAMA_CPP_SRC={p} does not exist.\n\
                 \n\
                 If you have a pre-built llama.cpp install use LLAMA_INSTALL_DIR instead:\n\
                   LLAMA_INSTALL_DIR={p} cargo build --features llama-cpp-static\n\
                 \n\
                 To build from source, clone llama.cpp first:\n\
                   git clone https://github.com/ggerganov/llama.cpp.git \"{p}\"\n"
            );
        }
        if !path.join("CMakeLists.txt").exists() {
            let has_binaries = path.read_dir().ok().map(|d| d.flatten().any(|e| {
                let ext = e.path().extension().map(|x| x.to_string_lossy().to_lowercase());
                matches!(ext.as_deref(), Some("so") | Some("dylib") | Some("dll") | Some("lib"))
            })).unwrap_or(false);
            if has_binaries {
                panic!(
                    "\nLLAMA_CPP_SRC={p} looks like a pre-built install, not a source tree.\n\
                     Use LLAMA_INSTALL_DIR instead:\n\
                       LLAMA_INSTALL_DIR={p} cargo build --features llama-cpp-static\n"
                );
            }
            panic!(
                "\nLLAMA_CPP_SRC={p} exists but has no CMakeLists.txt.\n\
                 Point it at a proper llama.cpp source checkout.\n"
            );
        }
        println!("cargo:rerun-if-changed={p}/CMakeLists.txt");
        return path;
    }

    // 2. Git submodule at ./llama.cpp (preferred for reproducible builds).
    let submodule = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set")
    ).join("llama.cpp");
    if submodule.join("CMakeLists.txt").exists() {
        println!("cargo:rerun-if-changed=llama.cpp/CMakeLists.txt");
        return submodule;
    }

    // 3. Auto-clone into OUT_DIR (works in CI without pre-cloning).
    let clone_target = out_dir.join("llama.cpp-src");
    if !clone_target.join("CMakeLists.txt").exists() {
        eprintln!("[build.rs] No local llama.cpp source — cloning from GitHub …");
        eprintln!("[build.rs] Tip: set LLAMA_INSTALL_DIR if you have a pre-built install.");
        let status = Command::new("git")
            .args([
                "clone", "--depth=1", "--branch", "master",
                "https://github.com/ggerganov/llama.cpp.git",
                clone_target.to_str().expect("non-UTF8 OUT_DIR"),
            ])
            .status()
            .expect("git clone failed — ensure git is in PATH");
        assert!(status.success(), "git clone llama.cpp failed");
    }
    clone_target
}

fn cmake_configure(src: &Path, build: &Path) {
    let cache_file = build.join("CMakeCache.txt");
    if cache_file.exists() {
        if let Ok(content) = std::fs::read_to_string(&cache_file) {
            let src_str = src.to_string_lossy();
            if !content.contains(&*src_str) {
                let _ = std::fs::remove_dir_all(build);
            }
        }
    }
    std::fs::create_dir_all(build).expect("create CMake build dir");

    let vulkan = std::env::var("LLAMA_VULKAN")
        .unwrap_or_else(|_| std::env::var("CARGO_FEATURE_VULKAN").unwrap_or_default());
    let cuda = std::env::var("LLAMA_CUDA")
        .unwrap_or_else(|_| std::env::var("CARGO_FEATURE_CUDA").unwrap_or_default());

    // Generator selection:
    let cmake_gen: &str = if cfg!(windows) && std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() == "msvc" {
        // Empty string forces CMake to auto-detect the newest Visual Studio generator
        // instead of falling back to Ninja (which might pick up MinGW gcc).
        ""
    } else if cmd_exists("ninja") {
        "Ninja"
    } else {
        "Unix Makefiles"
    };

    let mut cmd = Command::new("cmake");
    cmd.current_dir(build).arg(src);

    if !cmake_gen.is_empty() {
        cmd.arg(format!("-G{cmake_gen}"));
    }

    cmd // Static libraries on all platforms
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DGGML_SHARED_LIBS=OFF")
        // Disable all application targets (avoids missing -lllama-server-impl etc.)
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .arg("-DLLAMA_BUILD_SERVER=OFF")
        .arg("-DLLAMA_BUILD_TOOLS=OFF")
        .arg("-DLLAMA_STANDALONE=OFF");

    // PIC: needed on Unix for linking into a Rust binary; harmless on Windows.
    #[cfg(unix)]
    cmd.arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON");

    if cuda == "1" || cuda.eq_ignore_ascii_case("on") {
        cmd.arg("-DGGML_CUDA=ON");
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            cmd.arg(format!("-DCUDAToolkit_ROOT={}", cuda_path));
        }
    }
    if vulkan == "1" || vulkan.eq_ignore_ascii_case("on") {
        cmd.arg("-DGGML_VULKAN=ON");
    }

    let status = cmd.status().expect("cmake failed — is cmake installed?");
    assert!(status.success(), "CMake configure step failed");
}

fn cmake_build_libs(build: &Path) {
    let jobs = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "4".to_string());

    let mut args = vec![
        "--build".to_string(),
        ".".to_string(),
        "--config".to_string(),
        "Release".to_string(),
        "--parallel".to_string(),
        jobs,
    ];
    
    // By explicitly targeting libraries, we avoid building `llama-app` 
    // which has a known Ninja dependency race condition for build-info.h
    for t in ["llama", "ggml", "ggml-base"] {
        args.push("--target".to_string());
        args.push(t.to_string());
    }

    let status = Command::new("cmake")
        .args(&args)
        .current_dir(build)
        .status()
        .unwrap_or_else(|e| panic!("cmake --build failed: {e}"));

    // Note: ggml-base might be interface only in very old versions, but 
    // we require it for modern llama.cpp. If it fails, let it crash.
    assert!(status.success(), "cmake --build failed for static libraries");
}

fn emit_link_directives(build: &Path) {
    let archives = find_static_archives(build);
    assert!(
        !archives.is_empty(),
        "No static archives found under {} — did CMake build succeed?",
        build.display()
    );

    // Deduplicate lib dirs
    let mut lib_dirs: Vec<PathBuf> = Vec::new();
    for a in &archives {
        let d = a.parent().unwrap().to_path_buf();
        if !lib_dirs.contains(&d) { lib_dirs.push(d); }
    }
    for d in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }

    // Emit in correct link order
    let sorted = sort_libs(archives);
    for a in &sorted {
        let stem = a.file_stem().unwrap_or_default().to_string_lossy();
        // Unix: libllama.a → strip "lib"; Windows: llama.lib → no strip needed
        let name = stem.strip_prefix("lib").unwrap_or(&stem);
        println!("cargo:rustc-link-lib=static={name}");
    }

    link_system_libs();
}

/// Recursively collect static archives (.a on Unix, .lib on Windows),
/// skipping CMake internal directories.
fn find_static_archives(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            let n = p.file_name().unwrap_or_default().to_string_lossy();
            if matches!(n.as_ref(), "CMakeFiles" | "_deps" | "Testing" | "Release" | "Debug") {
                // On Windows, CMake puts the actual .lib files inside Release/ or Debug/
                // sub-dirs, so we DO recurse into Release/ but skip CMakeFiles etc.
                if n == "CMakeFiles" || n == "_deps" || n == "Testing" { continue; }
            }
            out.extend(find_static_archives(&p));
        } else {
            let ext = p.extension().map(|e| e.to_string_lossy().to_lowercase());
            // .a  → Unix/MinGW static lib
            // .lib → MSVC static lib  (but NOT import libs — those are for .dll)
            //   Heuristic: if a matching .dll exists, it's an import lib → skip it.
            //   Otherwise treat it as a static lib.
            match ext.as_deref() {
                Some("a") => out.push(p),
                Some("lib") => {
                    let dll = p.with_extension("dll");
                    if !dll.exists() { out.push(p); }
                }
                _ => {}
            }
        }
    }
    out
}

// ===========================================================================
// System libraries
// ===========================================================================

fn link_system_libs() {
    // ── C++ standard library ─────────────────────────────────────────────────
    // MSVC: auto-linked via #pragma comment(lib, ...) in the CRT headers.
    // MinGW / Linux: must be explicit.
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    println!("cargo:rustc-link-lib=stdc++");
    // MinGW on Windows also needs stdc++ (MSVC links it automatically)
    #[cfg(all(target_os = "windows", target_env = "gnu"))]
    println!("cargo:rustc-link-lib=stdc++");

    // ── POSIX threading ───────────────────────────────────────────────────────
    #[cfg(unix)]
    println!("cargo:rustc-link-lib=pthread");

    // ── Math / DL ─────────────────────────────────────────────────────────────
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }

    // ── CUDA runtime ────────────────────────────────────────────────────────
    let cuda = std::env::var("LLAMA_CUDA")
        .unwrap_or_else(|_| std::env::var("CARGO_FEATURE_CUDA").unwrap_or_default());
    if cuda == "1" || cuda.eq_ignore_ascii_case("on") {
        if let Ok(cuda_path) = std::env::var("CUDA_PATH").or_else(|_| std::env::var("CUDA_HOME")) {
            #[cfg(target_os = "windows")]
            println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
            #[cfg(not(target_os = "windows"))]
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        }
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
    }

    // ── Vulkan runtime ──────────────────────────────────────────────────────
    let vulkan = std::env::var("LLAMA_VULKAN")
        .unwrap_or_else(|_| std::env::var("CARGO_FEATURE_VULKAN").unwrap_or_default());
    if vulkan == "1" || vulkan.eq_ignore_ascii_case("on") {
        // We might also need a search path for VULKAN_SDK, if it's set
        if let Ok(vk_sdk) = std::env::var("VULKAN_SDK") {
            #[cfg(target_os = "windows")]
            println!("cargo:rustc-link-search=native={}/Lib", vk_sdk);
            #[cfg(not(target_os = "windows"))]
            println!("cargo:rustc-link-search=native={}/lib", vk_sdk);
        }
        
        #[cfg(target_os = "windows")]
        println!("cargo:rustc-link-lib=vulkan-1");
        #[cfg(not(target_os = "windows"))]
        println!("cargo:rustc-link-lib=vulkan");
    }

    // ── OpenMP runtime ────────────────────────────────────────────────────────
    // ggml-cpu is compiled with -fopenmp / /openmp and references GOMP_* or
    // omp_* symbols.
    //
    //  Linux / MinGW:  libgomp  (GCC's OpenMP runtime, ships with gcc)
    //  macOS:          omp      (llvm-openmp from Homebrew: brew install libomp)
    //  Windows MSVC:   vcomp    (Visual C++ OpenMP runtime, part of MSVC)
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=gomp");

    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-search=native=/opt/homebrew/opt/libomp/lib");
        println!("cargo:rustc-link-lib=omp");
        // Metal backend requirements
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }



    #[cfg(all(target_os = "windows", not(target_env = "gnu")))]
    println!("cargo:rustc-link-lib=vcomp");  // MSVC OpenMP

    #[cfg(all(target_os = "windows", target_env = "gnu"))]
    println!("cargo:rustc-link-lib=gomp");   // MinGW OpenMP
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Cross-platform "is this binary on PATH?".
fn cmd_exists(bin: &str) -> bool {
    // `which` on Unix, `where` on Windows
    let checker = if cfg!(windows) { "where" } else { "which" };
    Command::new(checker)
        .arg(bin)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
