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
//        # auto-clone (needs internet on first build; pins an immutable
//        # commit by default, override with LLAMA_CPP_REV=<sha|tag>)
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

/// Link target triple components for THIS build. build.rs executes on the
/// HOST, so `cfg!(target_os)` would be wrong under cross-compilation —
/// always use these (cargo sets CARGO_CFG_* for the target).
fn target_os() -> String {
    std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_else(|_| std::env::consts::OS.to_string())
}
fn target_env() -> String {
    std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default()
}

fn link_from_install_dir(dir: &Path) {
    // llama.cpp core libraries we care about (ignores openvino, tbb, hwloc…)
    let core_names = [
        "llama",
        "ggml",
        "ggml-base",
        "ggml-cpu",
        "ggml-rpc",
        "ggml-cuda",
        "ggml-vulkan",
        "ggml-metal",
        "llama-common",
        "mtmd",
    ];

    println!("cargo:rustc-link-search=native={}", dir.display());

    let files: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.path().is_file())
                .map(|e| e.file_name().to_string_lossy().to_lowercase())
                .collect()
        })
        .unwrap_or_default();
    match plan_prebuilt_links(&target_os(), &target_env(), &files, &core_names) {
        Ok(plan) => {
            for w in &plan.warnings {
                eprintln!("[build.rs] warning: {w}");
            }
            if plan.static_mode {
                eprintln!(
                    "[build.rs] {} static archive(s) in {} — static linking",
                    plan.libs.len(),
                    dir.display()
                );
                for lib in &plan.libs {
                    println!("cargo:rustc-link-lib=static={}", lib.name);
                }
            } else {
                eprintln!(
                    "[build.rs] No static archives in {} — dynamic linking with baked rpath",
                    dir.display()
                );
                // Bake the directory so the binary finds the libs without PATH/LD_LIBRARY_PATH.
                bake_rpath(dir);
                for lib in &plan.libs {
                    println!("cargo:rustc-link-lib=dylib={}", lib.name);
                }
            }
        }
        Err(msg) => panic!(
            "\nLLAMA_INSTALL_DIR={} unusable: {msg}\n\
             Is this the correct directory?\n",
            dir.display()
        ),
    }

    link_system_libs();
}

/// One rustc link directive from the planner.
#[derive(Debug, PartialEq)]
struct PlannedLib {
    kind: LinkKind,
    /// rustc-link-lib stem (e.g. `llama`, `ggml-cpu`).
    name: String,
}

#[derive(Debug, PartialEq)]
enum LinkKind {
    Static,
    Dylib,
}

#[derive(Debug)]
struct PrebuiltPlan {
    static_mode: bool,
    libs: Vec<PlannedLib>,
    warnings: Vec<String>,
}

/// Pure link planner for a prebuilt install directory: given the TARGET
/// platform and the directory's filenames, decide static vs dynamic and
/// the exact rustc directives — or fail with an actionable message.
///
/// Toolchain rules encoded here (verified against linker requirements):
/// - MSVC links NEVER consume a bare `.dll`: it needs the import library
///   (`<stem>.lib`, CMake also accepts `<stem>.dll.lib` via rustc's
///   search). A `X.lib` next to `X.dll` is an IMPORT lib, not a static
///   archive — linking it as `static=` would silently link the DLL while
///   reporting "static linking".
/// - MinGW ld links `lib<stem>.dll.a` preferably, else a bare `<stem>.dll`
///   directly (documented ld auto-import search); plain `.a` without a
///   same-stem `.dll` is static.
/// - Unix/macOS: `.a` static, versioned `.so` / `.dylib` dynamic (stem
///   mapping handles `libllama.so.0.0.1` → `llama`).
/// Static archives win whenever present (previous behavior preserved).
fn plan_prebuilt_links(
    target_os: &str,
    target_env: &str,
    files: &[String],
    core_names: &[&str],
) -> Result<PrebuiltPlan, String> {
    fn has(files: &[String], name: &str) -> bool {
        files.iter().any(|f| f == name)
    }
    // Link order rank (dependents before dependencies).
    fn rank(stem: &str) -> u8 {
        if stem == "mtmd" {
            0
        } else if stem == "llama" {
            1
        } else if stem.starts_with("ggml-") {
            3
        } else if stem == "ggml" {
            4
        } else {
            2
        }
    }
    let mut warnings: Vec<String> = Vec::new();

    if target_os == "windows" {
        let msvc = target_env != "gnu";
        // Per-stem file inventory (lowercased names).
        let mut statik: Vec<String> = Vec::new();
        let mut shared: Vec<String> = Vec::new();
        for core in core_names {
            if msvc {
                // A `X.lib` next to `X.dll` is an IMPORT lib, not a static
                // archive; a bare `X.lib` with no `X.dll` is static.
                let has_lib = has(files, &format!("{core}.lib"));
                let has_dll = has(files, &format!("{core}.dll"));
                let has_dll_lib = has(files, &format!("{core}.dll.lib"));
                if has_lib && !has_dll {
                    statik.push(core.to_string());
                } else if has_lib || has_dll_lib {
                    shared.push(core.to_string());
                } else if has_dll {
                    return Err(format!(
                        "`{core}.dll` found but no `{core}.lib` import library beside it — \
                         MSVC cannot link a bare .dll. Install the import library \
                         (CMake shared builds emit it next to the .dll)."
                    ));
                }
            } else {
                // MinGW: `libX.dll.a` preferred, bare `X.dll` via auto-import.
                let dll_a = files
                    .iter()
                    .any(|f| f == &format!("lib{core}.dll.a") || f == &format!("{core}.dll.a"));
                let bare_dll = has(files, &format!("{core}.dll"));
                let bare_a = (has(files, &format!("lib{core}.a"))
                    || has(files, &format!("{core}.a")))
                    && !bare_dll;
                if bare_a {
                    statik.push(core.to_string());
                } else if dll_a || bare_dll {
                    if bare_dll && !dll_a {
                        warnings.push(format!(
                            "`{core}.dll` without `lib{core}.dll.a`: relying on MinGW ld \
                             direct-DLL auto-import; prefer the import library."
                        ));
                    }
                    shared.push(core.to_string());
                }
            }
        }
        if !statik.is_empty() {
            statik.sort_by_key(|s| rank(s));
            return Ok(PrebuiltPlan {
                static_mode: true,
                libs: statik
                    .into_iter()
                    .map(|name| PlannedLib {
                        kind: LinkKind::Static,
                        name,
                    })
                    .collect(),
                warnings,
            });
        }
        if !shared.is_empty() {
            shared.sort_by_key(|s| rank(s));
            return Ok(PrebuiltPlan {
                static_mode: false,
                libs: shared
                    .into_iter()
                    .map(|name| PlannedLib {
                        kind: LinkKind::Dylib,
                        name,
                    })
                    .collect(),
                warnings,
            });
        }
        return Err(
            "has no linkable llama/ggml libraries (.lib with .dll, or static .lib/.a)".to_string(),
        );
    }

    // Unix/macOS: previous behavior (static .a wins; else versioned .so/.dylib).
    // Candidate filter mirrors the old collect_libs (lib*/llama*/ggml*).
    let is_candidate =
        |f: &String| f.starts_with("lib") || f.starts_with("llama") || f.starts_with("ggml");
    let mut archives: Vec<String> = files
        .iter()
        .filter(|f| f.ends_with(".a") && is_candidate(f))
        .cloned()
        .collect();
    archives.sort_by_key(|f| rank(&lib_stem(&PathBuf::from(f))));
    archives.dedup_by_key(|f| lib_stem(Path::new(f)));
    let shared_ext = shared_ext_for(target_os);
    let mut shared: Vec<String> = files
        .iter()
        .filter(|f| {
            is_candidate(f)
                && (f.ends_with(&format!(".{shared_ext}"))
                    || f.contains(&format!(".{shared_ext}.")))
        })
        .cloned()
        .collect();
    shared.sort_by_key(|f| rank(&lib_stem(&PathBuf::from(f))));
    shared.dedup_by_key(|f| lib_stem(Path::new(f)));
    // Core-name filtering happens BEFORE the emptiness decision: a
    // directory containing only unrelated archives (e.g. libunrelated.a)
    // must error, never produce a successful plan with zero libraries.
    let archives: Vec<PlannedLib> = archives
        .into_iter()
        .filter_map(|f| {
            let stem = lib_stem(&PathBuf::from(&f));
            core_names.contains(&stem.as_str()).then(|| PlannedLib {
                kind: LinkKind::Static,
                name: stem,
            })
        })
        .collect();
    let shared: Vec<PlannedLib> = shared
        .into_iter()
        .filter_map(|f| {
            let stem = lib_stem(&PathBuf::from(&f));
            core_names.contains(&stem.as_str()).then(|| PlannedLib {
                kind: LinkKind::Dylib,
                name: stem,
            })
        })
        .collect();
    if !archives.is_empty() {
        return Ok(PrebuiltPlan {
            static_mode: true,
            libs: archives,
            warnings,
        });
    }
    if !shared.is_empty() {
        return Ok(PrebuiltPlan {
            static_mode: false,
            libs: shared,
            warnings,
        });
    }
    Err("has no libllama.{a,so,dylib,lib,dll} files".to_string())
}

/// Bake the library directory into the binary so it is found at runtime
/// without the user setting LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
fn bake_rpath(dir: &Path) {
    let d = dir.to_string_lossy();
    // NOTE: target-gated at RUNTIME (build.rs runs on the host).
    if target_os() == "linux" || target_os() == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{d}");
    } else {
        // Windows uses PATH; there's no ELF rpath equivalent.
        // The DLLs must be alongside the .exe or in a directory on PATH.
        // We emit a warning so the packager knows.
        eprintln!(
            "[build.rs] Windows: DLLs from {d} must be in the same directory as hercules.exe \
             or on the system PATH at runtime."
        );
    }
}

/// Shared library extension for a TARGET platform (pure function so the
/// link matrix stays unit-testable; never `cfg!`, which answers for the
/// host).
fn shared_ext_for(target_os: &str) -> &'static str {
    if target_os == "windows" {
        "dll"
    } else if target_os == "macos" {
        "dylib"
    } else {
        "so"
    }
}

/// Extract the logical library name from a path.
///   libllama.so.0.0.1 → llama
///   llama.lib          → llama
///   libggml-cpu.a      → ggml-cpu
fn lib_stem(p: &Path) -> String {
    let fname = p
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    // Strip "lib" prefix if present
    let s = fname.strip_prefix("lib").unwrap_or(&fname);
    // Take everything before the first '.'
    s.split('.').next().unwrap_or(s).to_string()
}

fn sort_libs(mut libs: Vec<PathBuf>) -> Vec<PathBuf> {
    libs.sort_by_key(|p| {
        let s = lib_stem(p);
        if s == "mtmd" {
            0u8
        } else if s == "llama" {
            1
        } else if s.starts_with("ggml-") {
            3
        } else if s == "ggml" {
            4
        } else {
            2
        }
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
            let has_binaries = path
                .read_dir()
                .ok()
                .map(|d| {
                    d.flatten().any(|e| {
                        let ext = e
                            .path()
                            .extension()
                            .map(|x| x.to_string_lossy().to_lowercase());
                        matches!(
                            ext.as_deref(),
                            Some("so") | Some("dylib") | Some("dll") | Some("lib")
                        )
                    })
                })
                .unwrap_or(false);
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
    let submodule =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"))
            .join("llama.cpp");
    if submodule.join("CMakeLists.txt").exists() {
        println!("cargo:rerun-if-changed=llama.cpp/CMakeLists.txt");
        return submodule;
    }

    // 3. Auto-clone into OUT_DIR (works in CI without pre-cloning).
    // Reproducible: checks out an IMMUTABLE revision, never a moving
    // branch. Override with LLAMA_CPP_REV=<full-sha|tag>; the default is
    // pinned (master as of 2026-09-17) so clean builds are deterministic.
    const DEFAULT_LLAMA_CPP_REV: &str = "7f6f0c2a9dab36fdb1f6e00e2037c030974e0e5c";
    let rev = std::env::var("LLAMA_CPP_REV").unwrap_or_else(|_| DEFAULT_LLAMA_CPP_REV.to_string());
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_REV");
    let clone_target = out_dir.join("llama.cpp-src");
    if !clone_target.join("CMakeLists.txt").exists() {
        eprintln!("[build.rs] No local llama.cpp source — cloning {rev} from GitHub …");
        eprintln!("[build.rs] Tip: set LLAMA_INSTALL_DIR if you have a pre-built install.");
        let target_str = clone_target.to_str().expect("non-UTF8 OUT_DIR");
        let status = Command::new("git")
            .args([
                "clone",
                "https://github.com/ggerganov/llama.cpp.git",
                target_str,
            ])
            .status()
            .expect("git clone failed — ensure git is in PATH");
        assert!(status.success(), "git clone llama.cpp failed");
        let status = Command::new("git")
            .args(["-C", target_str, "checkout", rev.as_str()])
            .status()
            .expect("git checkout failed");
        assert!(
            status.success(),
            "git checkout {rev} failed (set LLAMA_CPP_REV to a valid SHA/tag)"
        );
    }
    // Verify the tree is at the requested revision (strict for SHAs).
    let head_out = Command::new("git")
        .args([
            "-C",
            clone_target.to_str().expect("non-UTF8 OUT_DIR"),
            "rev-parse",
            "HEAD",
        ])
        .output();
    if let Ok(out) = head_out {
        let head = String::from_utf8_lossy(&out.stdout).trim().to_string();
        eprintln!("[build.rs] llama.cpp source @ {head} (requested {rev})");
        let looks_like_sha = rev.len() >= 7 && rev.chars().all(|c| c.is_ascii_hexdigit());
        if looks_like_sha {
            assert!(
                head.starts_with(&rev),
                "llama.cpp tree is at {head}, expected {rev} — delete {} and rebuild",
                clone_target.display()
            );
        }
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

    // Generator selection (target-aware: build.rs runs on the host).
    let cmake_gen: &str = if cmd_exists("ninja") {
        "Ninja"
    } else if target_os() == "windows" {
        ""
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
        .arg("-DLLAMA_BUILD_MTMD=ON")
        .arg("-DLLAMA_STANDALONE=OFF");

    // PIC: needed on Unix for linking into a Rust binary; harmless on Windows.
    // Target-gated (build.rs runs on the host).
    if target_os() != "windows" {
        cmd.arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON");
    }

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
    for t in ["llama", "ggml", "ggml-base", "mtmd"] {
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
    assert!(
        status.success(),
        "cmake --build failed for static libraries"
    );
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
        if !lib_dirs.contains(&d) {
            lib_dirs.push(d);
        }
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
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            let n = p.file_name().unwrap_or_default().to_string_lossy();
            if matches!(
                n.as_ref(),
                "CMakeFiles" | "_deps" | "Testing" | "Release" | "Debug"
            ) {
                // On Windows, CMake puts the actual .lib files inside Release/ or Debug/
                // sub-dirs, so we DO recurse into Release/ but skip CMakeFiles etc.
                if n == "CMakeFiles" || n == "_deps" || n == "Testing" {
                    continue;
                }
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
                    if !dll.exists() {
                        out.push(p);
                    }
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
    // All branches below are TARGET-gated at runtime (build.rs runs on
    // the host; `cfg!` would answer for the wrong platform when
    // cross-compiling).
    let os = target_os();
    let env = target_env();
    // ── C++ standard library ─────────────────────────────────────────────────
    // MSVC: auto-linked via #pragma comment(lib, ...) in the CRT headers.
    // MinGW / Linux: must be explicit.
    if os == "macos" {
        println!("cargo:rustc-link-lib=c++");
    }
    if os != "macos" && os != "windows" {
        println!("cargo:rustc-link-lib=stdc++");
    }
    // MinGW on Windows also needs stdc++ (MSVC links it automatically)
    if os == "windows" && env == "gnu" {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // ── POSIX threading ───────────────────────────────────────────────────────
    if os != "windows" {
        println!("cargo:rustc-link-lib=pthread");
    }

    // ── Math / DL ─────────────────────────────────────────────────────────────
    if os == "linux" {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }

    // ── CUDA runtime ────────────────────────────────────────────────────────
    let cuda = std::env::var("LLAMA_CUDA")
        .unwrap_or_else(|_| std::env::var("CARGO_FEATURE_CUDA").unwrap_or_default());
    if cuda == "1" || cuda.eq_ignore_ascii_case("on") {
        if let Ok(cuda_path) = std::env::var("CUDA_PATH").or_else(|_| std::env::var("CUDA_HOME")) {
            if os == "windows" {
                println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
            } else {
                println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
                println!("cargo:rustc-link-search=native={}/lib64/stubs", cuda_path);
            }
        }
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");
        println!("cargo:rustc-link-lib=cuda");
    }

    // ── Vulkan runtime ──────────────────────────────────────────────────────
    let vulkan = std::env::var("LLAMA_VULKAN")
        .unwrap_or_else(|_| std::env::var("CARGO_FEATURE_VULKAN").unwrap_or_default());
    if vulkan == "1" || vulkan.eq_ignore_ascii_case("on") {
        // We might also need a search path for VULKAN_SDK, if it's set
        if let Ok(vk_sdk) = std::env::var("VULKAN_SDK") {
            if os == "windows" {
                println!("cargo:rustc-link-search=native={}/Lib", vk_sdk);
            } else {
                println!("cargo:rustc-link-search=native={}/lib", vk_sdk);
            }
        }

        if os == "windows" {
            println!("cargo:rustc-link-lib=vulkan-1");
        } else {
            println!("cargo:rustc-link-lib=vulkan");
        }
    }

    // ── OpenMP runtime ────────────────────────────────────────────────────────
    // ggml-cpu is compiled with -fopenmp / /openmp and references GOMP_* or
    // omp_* symbols.
    //
    //  Linux / MinGW:  libgomp  (GCC's OpenMP runtime, ships with gcc)
    //  macOS:          omp      (llvm-openmp from Homebrew: brew install libomp)
    //  Windows MSVC:   vcomp    (Visual C++ OpenMP runtime, part of MSVC)
    if os == "linux" {
        println!("cargo:rustc-link-lib=gomp");
    }

    if os == "macos" {
        println!("cargo:rustc-link-search=native=/opt/homebrew/opt/libomp/lib");
        println!("cargo:rustc-link-lib=omp");
        // Metal backend requirements
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    if os == "windows" && env != "gnu" {
        println!("cargo:rustc-link-lib=vcomp"); // MSVC OpenMP
    }

    if os == "windows" && env == "gnu" {
        println!("cargo:rustc-link-lib=gomp"); // MinGW OpenMP
    }
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

// ===========================================================================
// Prebuilt link matrix tests (run: rustc --test build.rs -o /tmp/brs && /tmp/brs)
// ===========================================================================

#[cfg(test)]
mod prebuilt_matrix_tests {
    use super::*;

    const CORE: &[&str] = &["llama", "ggml", "ggml-base", "ggml-cpu"];

    fn names(files: &[&str]) -> Vec<String> {
        files.iter().map(|s| s.to_ascii_lowercase()).collect()
    }

    fn kinds(plan: &PrebuiltPlan) -> Vec<(&str, &LinkKind)> {
        plan.libs
            .iter()
            .map(|l| (l.name.as_str(), &l.kind))
            .collect()
    }

    #[test]
    fn msvc_static_lib_preferred() {
        // llama.lib + ggml.lib, no DLLs → static mode.
        let plan = plan_prebuilt_links(
            "windows",
            "msvc",
            &names(&["llama.lib", "ggml.lib", "ggml-cpu.lib", "readme.txt"]),
            CORE,
        )
        .expect("static plan");
        assert!(plan.static_mode);
        assert!(kinds(&plan).contains(&("llama", &LinkKind::Static)));
    }

    #[test]
    fn msvc_import_lib_means_shared() {
        // llama.lib NEXT TO llama.dll is an import lib, not static:
        // must plan dynamic, never claim "static linking".
        let plan = plan_prebuilt_links(
            "windows",
            "msvc",
            &names(&["llama.lib", "llama.dll", "ggml.lib", "ggml.dll"]),
            CORE,
        )
        .expect("shared plan");
        assert!(!plan.static_mode);
        assert!(kinds(&plan).contains(&("llama", &LinkKind::Dylib)));
    }

    #[test]
    fn msvc_bare_dll_is_actionable_error() {
        // A bare .dll cannot satisfy MSVC link.exe: fail with guidance,
        // never emit a doomed dylib= directive.
        let err = plan_prebuilt_links("windows", "msvc", &names(&["llama.dll"]), CORE)
            .expect_err("must refuse bare dll");
        assert!(err.contains("import library"), "got: {err}");
    }

    #[test]
    fn msvc_dll_dot_lib_variant_accepted() {
        // CMake-style `llama.dll.lib` import library satisfies the link.
        let plan = plan_prebuilt_links(
            "windows",
            "msvc",
            &names(&["llama.dll.lib", "llama.dll"]),
            CORE,
        )
        .expect("shared plan");
        assert!(!plan.static_mode);
    }

    #[test]
    fn mingw_dll_import_lib_preferred() {
        let plan = plan_prebuilt_links(
            "windows",
            "gnu",
            &names(&["libllama.dll.a", "libggml.dll.a"]),
            CORE,
        )
        .expect("shared plan");
        assert!(!plan.static_mode);
        assert!(kinds(&plan).contains(&("llama", &LinkKind::Dylib)));
    }

    #[test]
    fn mingw_static_wins_when_both_present() {
        // A real static archive alongside an import lib keeps the
        // historical static-first precedence.
        let plan = plan_prebuilt_links(
            "windows",
            "gnu",
            &names(&["libllama.a", "libllama.dll.a"]),
            CORE,
        )
        .expect("static plan");
        assert!(plan.static_mode);
    }

    #[test]
    fn mingw_bare_dll_links_with_warning() {
        // MinGW ld auto-import can consume a bare .dll directly.
        let plan = plan_prebuilt_links("windows", "gnu", &names(&["llama.dll"]), CORE)
            .expect("shared plan");
        assert!(!plan.static_mode);
        assert!(!plan.warnings.is_empty(), "must warn about auto-import");
    }

    #[test]
    fn mingw_static_without_dll() {
        let plan = plan_prebuilt_links("windows", "gnu", &names(&["libllama.a"]), CORE)
            .expect("static plan");
        assert!(plan.static_mode);
    }

    #[test]
    fn linux_versioned_so_dynamic() {
        let plan = plan_prebuilt_links(
            "linux",
            "",
            &names(&["libllama.so.0.0.1", "libggml.so.0"]),
            CORE,
        )
        .expect("shared plan");
        assert!(!plan.static_mode);
        assert!(kinds(&plan).contains(&("llama", &LinkKind::Dylib)));
    }

    #[test]
    fn linux_static_wins() {
        let plan = plan_prebuilt_links("linux", "", &names(&["libllama.a", "libllama.so"]), CORE)
            .expect("static plan");
        assert!(plan.static_mode);
    }

    #[test]
    fn macos_dylib() {
        let plan = plan_prebuilt_links("macos", "", &names(&["libllama.dylib"]), CORE)
            .expect("shared plan");
        assert!(!plan.static_mode);
    }

    #[test]
    fn empty_dir_is_actionable_error() {
        let err = plan_prebuilt_links("linux", "", &[], CORE).expect_err("must refuse");
        assert!(err.contains("libllama"), "got: {err}");
    }

    #[test]
    fn linux_unrelated_archive_is_rejected() {
        // Unrelated archives satisfy the broad candidate predicate but
        // must not produce a successful plan with zero libraries.
        let err = plan_prebuilt_links("linux", "", &names(&["libunrelated.a"]), CORE)
            .expect_err("unrelated archive must not produce empty successful plan");
        assert!(err.contains("llama"), "got: {err}");
    }

    #[test]
    fn linux_unrelated_shared_library_is_rejected() {
        let err = plan_prebuilt_links("linux", "", &names(&["libunrelated.so"]), CORE)
            .expect_err("unrelated shared lib must not produce empty successful plan");
        assert!(err.contains("llama"), "got: {err}");
    }

    #[test]
    fn link_order_llama_before_ggml() {
        // Dependents precede dependencies for static archives.
        let plan = plan_prebuilt_links(
            "linux",
            "",
            &names(&["libggml.a", "libllama.a", "libggml-cpu.a"]),
            CORE,
        )
        .expect("static plan");
        let order: Vec<&str> = plan.libs.iter().map(|l| l.name.as_str()).collect();
        let llama = order.iter().position(|n| *n == "llama").unwrap();
        let ggml = order.iter().position(|n| *n == "ggml").unwrap();
        assert!(llama < ggml, "order: {order:?}");
    }
}
