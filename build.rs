// build.rs — links llama.cpp into the Hercules binary at build time so the
// final executable is self-contained (no runtime library search required).
//
// Two modes, tried in order:
//
//   A) Pre-built install  (LLAMA_INSTALL_DIR)
//      Point at a directory that already contains .so / .a files.
//      If .a files are found   → static link (truly self-contained).
//      If only .so files found → dynamic link + bake RPATH so the binary
//                                finds the libs without LD_LIBRARY_PATH.
//      Example:
//        LLAMA_INSTALL_DIR=/opt/llama.cpp cargo build --features llama-cpp-static
//
//   B) Build from source  (LLAMA_CPP_SRC / submodule / auto-clone)
//      Requires cmake + a C++17 compiler.  Produces static .a files.
//      Examples:
//        # submodule (recommended for reproducible builds)
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
// dlopen path is used and this script does nothing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_INSTALL_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_SRC");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_VULKAN");

    if std::env::var("CARGO_FEATURE_LLAMA_CPP_STATIC").is_err() {
        return; // runtime dlopen path; nothing to do
    }

    // ── Mode A: pre-built install directory ──────────────────────────────────
    if let Ok(dir) = std::env::var("LLAMA_INSTALL_DIR") {
        let path = PathBuf::from(&dir);
        if !path.exists() {
            panic!(
                "\nLLAMA_INSTALL_DIR={dir} does not exist.\n\
                 Set it to the directory that contains libllama.so / libllama.a.\n"
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
    // Collect all relevant library files in the directory.
    let archives = collect_libs(dir, "a");
    let shared   = collect_libs(dir, shared_ext());

    // Filter to just the llama.cpp / ggml libraries we actually need.
    // Ignore openvino, tbb, hwloc etc. — they'll be pulled in transitively.
    let core_names = [
        "llama", "ggml", "ggml-base", "ggml-cpu",
        "ggml-rpc", "ggml-cuda", "ggml-vulkan", "ggml-metal",
        "llama-common",
    ];

    println!("cargo:rustc-link-search=native={}", dir.display());

    if !archives.is_empty() {
        // Static archives available — prefer them.
        eprintln!("[build.rs] Found {} static archive(s) in {} — static linking", archives.len(), dir.display());
        let sorted = sort_libs(archives);
        for path in &sorted {
            let stem = lib_stem(&path);
            if core_names.iter().any(|n| stem == *n) {
                println!("cargo:rustc-link-lib=static={stem}");
            }
        }
    } else if !shared.is_empty() {
        // Only shared libraries — dynamic link + bake RPATH so the binary
        // finds them without the user setting LD_LIBRARY_PATH.
        eprintln!(
            "[build.rs] No static archives found in {} — dynamic linking with baked RPATH",
            dir.display()
        );
        let rpath = dir.to_string_lossy();
        // Bake the install dir as an RPATH into the final binary.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{rpath}");

        let sorted = sort_libs(shared);
        for path in &sorted {
            let stem = lib_stem(&path);
            if core_names.iter().any(|n| stem == *n) {
                println!("cargo:rustc-link-lib=dylib={stem}");
            }
        }
    } else {
        panic!(
            "\nLLAMA_INSTALL_DIR={} contains no libllama.{{a,so}} files.\n\
             Is this the right directory?  Contents must include libllama.so or libllama.a.\n",
            dir.display()
        );
    }

    link_system_libs();
}

fn collect_libs(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file() && p.extension().map(|e| e == ext).unwrap_or(false) {
            // Only pick up files whose name starts with "lib"
            let fname = p.file_name().unwrap_or_default().to_string_lossy();
            if fname.starts_with("lib") {
                out.push(p);
            }
        }
    }
    out
}

#[cfg(target_os = "macos")]
fn shared_ext() -> &'static str { "dylib" }
#[cfg(not(target_os = "macos"))]
fn shared_ext() -> &'static str { "so" }

fn lib_stem(p: &Path) -> String {
    // libllama.so.0.0.1 → llama   (take the first component after "lib")
    let fname = p.file_name().unwrap_or_default().to_string_lossy();
    // Strip "lib" prefix then take up to first '.'
    let without_prefix = fname.strip_prefix("lib").unwrap_or(&fname);
    without_prefix.split('.').next().unwrap_or(without_prefix).to_string()
}

fn sort_libs(mut libs: Vec<PathBuf>) -> Vec<PathBuf> {
    libs.sort_by_key(|p| {
        let s = lib_stem(p);
        if s == "llama"               { 0u8 }
        else if s.starts_with("ggml-") { 2 }
        else if s == "ggml"            { 3 }
        else                           { 1 }
    });
    // Deduplicate by stem (e.g. libllama.so / libllama.so.0 / libllama.so.0.0.1)
    let mut seen = std::collections::HashSet::new();
    libs.retain(|p| seen.insert(lib_stem(p)));
    libs
}

// ===========================================================================
// Mode B — build from source
// ===========================================================================

fn locate_or_fetch_source(out_dir: &Path) -> PathBuf {
    // 1. Explicit source tree.
    if let Ok(p) = std::env::var("LLAMA_CPP_SRC") {
        let path = PathBuf::from(&p);
        if !path.exists() {
            panic!(
                "\nLLAMA_CPP_SRC={p} does not exist.\n\
                 \n\
                 If you have a pre-built llama.cpp install (with .so files), use:\n\
                   LLAMA_INSTALL_DIR={p} cargo build --features llama-cpp-static\n\
                 \n\
                 To build from source, clone llama.cpp first:\n\
                   git clone https://github.com/ggerganov/llama.cpp.git {p}\n"
            );
        }
        if !path.join("CMakeLists.txt").exists() {
            // Might be a binary install — give a helpful hint.
            let has_so = path.read_dir()
                .ok()
                .map(|d| d.flatten().any(|e| {
                    e.path().extension().map(|x| x == "so" || x == "dylib" || x == "dll").unwrap_or(false)
                }))
                .unwrap_or(false);
            if has_so {
                panic!(
                    "\nLLAMA_CPP_SRC={p} looks like a pre-built install (has .so files) not a source tree.\n\
                     Use LLAMA_INSTALL_DIR instead:\n\
                       LLAMA_INSTALL_DIR={p} cargo build --features llama-cpp-static\n"
                );
            }
            panic!(
                "\nLLAMA_CPP_SRC={p} exists but has no CMakeLists.txt.\n\
                 Point LLAMA_CPP_SRC at a proper llama.cpp source checkout.\n"
            );
        }
        println!("cargo:rerun-if-changed={p}/CMakeLists.txt");
        return path;
    }

    // 2. Git submodule at ./llama.cpp.
    let submodule = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("llama.cpp");
    if submodule.join("CMakeLists.txt").exists() {
        println!("cargo:rerun-if-changed=llama.cpp/CMakeLists.txt");
        return submodule;
    }

    // 3. Auto-clone.
    let clone_target = out_dir.join("llama.cpp-src");
    if !clone_target.join("CMakeLists.txt").exists() {
        eprintln!("[build.rs] No local llama.cpp source found — cloning from GitHub …");
        eprintln!("[build.rs] Tip: set LLAMA_INSTALL_DIR=/opt/llama.cpp if you have a pre-built install.");
        let status = Command::new("git")
            .args([
                "clone", "--depth=1", "--branch", "master",
                "https://github.com/ggerganov/llama.cpp.git",
                clone_target.to_str().unwrap(),
            ])
            .status()
            .expect("git clone failed — ensure git is in PATH");
        assert!(status.success(), "git clone llama.cpp failed");
    }
    clone_target
}

fn cmake_configure(src: &Path, build: &Path) {
    std::fs::create_dir_all(build).expect("create build dir");

    let cuda   = std::env::var("LLAMA_CUDA").unwrap_or_default();
    let vulkan = std::env::var("LLAMA_VULKAN").unwrap_or_default();
    let cmake_gen = if which("ninja") { "Ninja" } else { "Unix Makefiles" };

    let mut cmd = Command::new("cmake");
    cmd.current_dir(build)
        .arg(src)
        .arg(format!("-G{cmake_gen}"))
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DGGML_SHARED_LIBS=OFF")
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .arg("-DLLAMA_BUILD_SERVER=OFF")
        .arg("-DLLAMA_BUILD_TOOLS=OFF")
        .arg("-DLLAMA_STANDALONE=OFF")
        .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON");

    if cuda == "1" || cuda.eq_ignore_ascii_case("on") {
        cmd.arg("-DGGML_CUDA=ON");
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

    for target in &["llama", "ggml"] {
        let status = Command::new("cmake")
            .args(["--build", ".", "--config", "Release", "--target", target, "--parallel", &jobs])
            .current_dir(build)
            .status()
            .unwrap_or_else(|e| panic!("cmake --build --target {target} failed: {e}"));

        if !status.success() && *target == "ggml" {
            eprintln!("[build.rs] cmake --target ggml skipped (may be interface-only)");
        } else {
            assert!(status.success(), "cmake --build --target {target} failed");
        }
    }
}

fn emit_link_directives(build: &Path) {
    let archives = find_archives(build);
    assert!(!archives.is_empty(), "No .a files found under {}", build.display());

    let mut lib_dirs: Vec<PathBuf> = Vec::new();
    for a in &archives {
        let d = a.parent().unwrap().to_path_buf();
        if !lib_dirs.contains(&d) { lib_dirs.push(d); }
    }
    for d in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", d.display());
    }

    let sorted = sort_libs(archives);
    for a in &sorted {
        let stem = a.file_stem().unwrap_or_default().to_string_lossy();
        let name = stem.strip_prefix("lib").unwrap_or(&stem);
        println!("cargo:rustc-link-lib=static={name}");
    }

    link_system_libs();
}

fn find_archives(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            let n = p.file_name().unwrap_or_default().to_string_lossy();
            if n == "CMakeFiles" || n == "_deps" || n == "Testing" { continue; }
            out.extend(find_archives(&p));
        } else if p.extension().map(|e| e == "a").unwrap_or(false) {
            out.push(p);
        }
    }
    out
}

// ===========================================================================
// Shared helpers
// ===========================================================================

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
    // ggml-cpu is compiled with -fopenmp; the resulting .a references GOMP_*
    // symbols from libgomp (GCC OpenMP runtime).  Always present alongside gcc.
    // macOS uses libomp from LLVM/Homebrew instead.
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=gomp");
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=omp");
}

fn which(bin: &str) -> bool {
    Command::new("which").arg(bin).output().map(|o| o.status.success()).unwrap_or(false)
}
