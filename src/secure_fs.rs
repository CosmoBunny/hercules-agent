//! TOCTOU-safe file opening bound to sandbox authorization.
//!
//! `path_allowed()` authorizes a NAME, but a symlink can be swapped between
//! that check and `open`, redirecting the read outside the CurrentDir
//! boundary. This module binds authorization to the OPENED OBJECT:
//!
//! - Linux: `open()` by name, then `readlink(/proc/self/fd/N)` resolves the
//!   ACTUAL opened object; it must lie beneath the sandbox root. A swap
//!   after `open` cannot change what the fd refers to; a swap before
//!   `open` is caught because verification runs on the opened object.
//! - macOS: same shape via `fcntl(F_GETPATH)` (libc is a direct dependency).
//! - Windows: `CreateFileW` open, then `GetFinalPathNameByHandleW`
//!   resolves the OPENED object (the OS strips every reparse point — the
//!   result is symlink-free `\\?\` form); required beneath the sandbox
//!   root via case-insensitive verbatim comparison. Same invariant:
//!   authorize → open → inspect handle → prove → read handle.
//!   (windows-sys is a Windows-only dependency for exactly these APIs.)
//! - Other platforms: `canonicalize` before open, open, `canonicalize`
//!   again — the two resolutions must be identical and beneath the root.
//!   Best-effort: std exposes no handle→path API there, so a
//!   continuous-race attacker theoretically retains a microsecond window.
//!   Documented, not hidden.
//!
//! Callers read through the returned handle and `fstat` the handle (never
//! re-resolve names), and cap the read itself so a racing grow cannot
//! overflow memory. Directory listings are intentionally out of scope
//! here (they need fd-based readdir, a separate primitive); `execute_ls`
//! re-verifies its path after listing as a best-effort narrowing step.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// An opened file whose object was verified beneath the sandbox root
/// (or opened unrestricted under AllDirs).
#[derive(Debug)]
pub struct SecureFile {
    pub file: File,
    /// Object path as observed at verification time (symlinks resolved).
    /// Informational: identity keys must still use `stable_key` upstream.
    pub verified_path: PathBuf,
}

#[derive(Debug)]
pub enum SecureOpenError {
    NotFound,
    SandboxDenied(String),
    /// Directory creation failed (carries OS detail for caller messages).
    MkdirFailed(String),
    /// Pre/post-open resolutions disagreed (non-fd platforms): something
    /// moved under the open. Never served.
    SwappedDuringOpen,
    Io(std::io::Error),
}

/// Open `path` for reading with object-bound sandbox verification.
/// `sandbox_root`: `Some(canonical cwd)` under CurrentDir scope, `None`
/// under AllDirs (unrestricted by the Hercules sandbox).
pub fn secure_open_read(
    path: &Path,
    sandbox_root: Option<&Path>,
) -> Result<SecureFile, SecureOpenError> {
    #[cfg(windows)]
    {
        return secure_open_read_windows(path, sandbox_root);
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        return secure_open_read_fallback(path, sandbox_root);
    }
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let file = File::open(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                SecureOpenError::NotFound
            } else {
                SecureOpenError::Io(e)
            }
        })?;
        let Some(root) = sandbox_root else {
            // AllDirs: no Hercules boundary to enforce.
            return Ok(SecureFile {
                verified_path: path.to_path_buf(),
                file,
            });
        };
        let object = object_path_of(&file)?;
        // Strip the Linux " (deleted)" marker for the boundary comparison
        // (the object itself was verified open and readable).
        let object_str = object.to_string_lossy();
        let object_trimmed = object_str
            .strip_suffix(" (deleted)")
            .map(PathBuf::from)
            .unwrap_or_else(|| object.clone());
        if !object_trimmed.starts_with(root) {
            return Err(SecureOpenError::SandboxDenied(format!(
                "Safefolder blocked path '{}' (opened object '{}' outside current dir {}). \
                 Menu → Permissions → Interact on all directories.",
                path.display(),
                object.display(),
                root.display()
            )));
        }
        Ok(SecureFile {
            verified_path: object,
            file,
        })
    }
}

/// Verify an OPEN handle's object lies beneath `root` (`None` under
/// AllDirs scope = unrestricted). Shared by the read and write paths so
/// authorization always targets the opened object, never a re-resolved
/// name. Returns the verified object path.
fn verify_handle_object(
    file: &File,
    root: Option<&Path>,
    display: &Path,
) -> Result<PathBuf, SecureOpenError> {
    let Some(root) = root else {
        return Ok(display.to_path_buf());
    };
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let object = object_path_of(file)?;
        // Strip the Linux " (deleted)" marker for the boundary comparison
        // (the object itself is verified open).
        let object_str = object.to_string_lossy();
        let trimmed = object_str
            .strip_suffix(" (deleted)")
            .map(PathBuf::from)
            .unwrap_or_else(|| object.clone());
        if !trimmed.starts_with(root) {
            return Err(SecureOpenError::SandboxDenied(format!(
                "Safefolder blocked path '{}' (opened object '{}' outside current dir {}). \
                 Menu → Permissions → Interact on all directories.",
                display.display(),
                object.display(),
                root.display()
            )));
        }
        Ok(object)
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::HANDLE;
        let object = final_path_of_handle(file.as_raw_handle() as HANDLE)?;
        if !verbatim_beneath(&object, root) {
            return Err(SecureOpenError::SandboxDenied(format!(
                "Safefolder blocked path '{}' (opened object '{}' outside current dir {}). \
                 Menu → Permissions → Interact on all directories.",
                display.display(),
                object.display(),
                root.display()
            )));
        }
        Ok(object)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        // No handle→path API: canonicalize-compare (best effort, narrowed
        // window — documented at secure_open_read_fallback).
        let current = display.canonicalize().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                SecureOpenError::NotFound
            } else {
                SecureOpenError::Io(e)
            }
        })?;
        if !current.starts_with(root) {
            return Err(SecureOpenError::SandboxDenied(format!(
                "Safefolder blocked path '{}'.",
                display.display()
            )));
        }
        Ok(current)
    }
}

/// Resolve the filesystem object behind an open handle.
/// Linux via /proc/self/fd (always available, no extra deps);
/// macOS via fcntl(F_GETPATH).
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn object_path_of(file: &File) -> Result<PathBuf, SecureOpenError> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let link = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
        std::fs::read_link(link).map_err(SecureOpenError::Io)
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::io::AsRawFd;
        let mut buf = vec![0u8; 1024];
        // SAFETY: F_GETPATH writes a NUL-terminated path into our buffer.
        let ret = unsafe {
            libc::fcntl(
                file.as_raw_fd(),
                libc::F_GETPATH,
                buf.as_mut_ptr() as *mut libc::c_char,
            )
        };
        if ret == -1 {
            return Err(SecureOpenError::Io(std::io::Error::last_os_error()));
        }
        let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(PathBuf::from(
            String::from_utf8_lossy(&buf[..len]).to_string(),
        ))
    }
}

/// Windows handle/object verification: open with CreateFileW, resolve the
/// OPENED object with GetFinalPathNameByHandleW (the OS resolves every
/// reparse point — the result contains no symlinks), and require it
/// beneath the sandbox root. A reparse swap before `open` is caught
/// because verification runs on the opened object; a swap after `open`
/// cannot redirect the handle. Bytes are read through the same handle,
/// satisfying: authorize → open → inspect handle → prove → read handle.
#[cfg(windows)]
fn secure_open_read_windows(
    path: &Path,
    sandbox_root: Option<&Path>,
) -> Result<SecureFile, SecureOpenError> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use std::os::windows::io::{FromRawHandle, OwnedHandle};
    use windows_sys::Win32::Foundation::{
        CloseHandle, GENERIC_READ, GetLastError, HANDLE, INVALID_HANDLE_VALUE,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_NAME_NORMALIZED, FILE_SHARE_DELETE,
        FILE_SHARE_READ, FILE_SHARE_WRITE, GetFinalPathNameByHandleW, OPEN_EXISTING,
    };

    const ERROR_FILE_NOT_FOUND: u32 = 2;
    const ERROR_PATH_NOT_FOUND: u32 = 3;

    let wide: Vec<u16> = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    // SAFETY: `wide` is NUL-terminated; all other args are valid constants
    // or null; the returned handle is checked before any use. The template
    // handle parameter takes null (no template file).
    let handle: HANDLE = unsafe {
        CreateFileW(
            wide.as_ptr(),
            GENERIC_READ,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            std::ptr::null(),
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        // SAFETY: just failed; GetLastError is valid here.
        let code = unsafe { GetLastError() };
        if code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND {
            return Err(SecureOpenError::NotFound);
        }
        return Err(SecureOpenError::Io(std::io::Error::from_raw_os_error(
            code as i32,
        )));
    }
    // Resolve the OPENED object (symlink-free \\?\ form). Query length first.
    // SAFETY: handle is valid; null buffer with length 0 is the documented
    // length-query call.
    let needed =
        unsafe { GetFinalPathNameByHandleW(handle, std::ptr::null_mut(), 0, FILE_NAME_NORMALIZED) };
    if needed == 0 {
        unsafe { CloseHandle(handle) };
        return Err(SecureOpenError::Io(std::io::Error::last_os_error()));
    }
    let mut buf: Vec<u16> = vec![0; needed as usize + 1];
    // SAFETY: buf has the queried capacity; handle is valid.
    let written = unsafe {
        GetFinalPathNameByHandleW(
            handle,
            buf.as_mut_ptr(),
            buf.len() as u32,
            FILE_NAME_NORMALIZED,
        )
    };
    if written == 0 || written as usize >= buf.len() {
        unsafe { CloseHandle(handle) };
        return Err(SecureOpenError::Io(std::io::Error::last_os_error()));
    }
    buf.truncate(written as usize);
    let object = PathBuf::from(std::ffi::OsString::from_wide(&buf));
    if let Some(root) = sandbox_root {
        if !verbatim_beneath(&object, root) {
            unsafe { CloseHandle(handle) };
            return Err(SecureOpenError::SandboxDenied(format!(
                "Safefolder blocked path '{}' (opened object '{}' outside current dir {}). \
                 Menu → Permissions → Interact on all directories.",
                path.display(),
                object.display(),
                root.display()
            )));
        }
    }
    // Transfer ownership: File closes the handle on drop (no leak on any
    // later path — every early return above already closed it).
    // SAFETY: handle is a valid owned file handle exactly once.
    let owned = unsafe { OwnedHandle::from_raw_handle(handle as *mut std::ffi::c_void) };
    Ok(SecureFile {
        verified_path: object,
        file: File::from(owned),
    })
}

/// Verbatim-path containment for `\\?\`-form Windows paths, compared
/// case-insensitively on a directory boundary (Windows paths are
/// case-insensitive; a trailing separator prevents `C:\foo` matching
/// `C:\foobar`).
#[cfg(windows)]
fn verbatim_beneath(object: &Path, root: &Path) -> bool {
    fn norm(p: &Path) -> String {
        p.as_os_str()
            .to_string_lossy()
            .to_ascii_lowercase()
            .replace('/', "\\")
    }
    let obj = norm(object);
    let mut base = norm(root);
    if !base.ends_with('\\') {
        base.push('\\');
    }
    obj == base.trim_end_matches('\\') || obj.starts_with(&base)
}

/// How a secure write open treats a missing target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOpenMode {
    /// Fail when the target does not exist (range replacement).
    OpenExisting,
    /// Create the target when missing (full-file write).
    OpenOrCreate,
}

/// A read+write handle to a verified object. All subsequent mutation
/// (truncate, write) MUST go through `file` — never re-resolve the name.
pub struct SecureWriteFile {
    pub file: File,
    pub verified_path: PathBuf,
    /// True when THIS open created the file. Used for rollback when
    /// post-open verification fails (removes our own creation; never
    /// touches pre-existing files).
    pub created_by_open: bool,
}

/// Open a file for handle-bound mutation: authorize → open → verify the
/// opened OBJECT → mutate the same handle. Never `fs::write()` by name
/// after this returns.
///
/// Safety properties:
/// - No `O_TRUNC`/truncate happens before verification (truncation would
///   itself be an unverified mutation); callers truncate via the handle
///   only after success.
/// - In create mode the file may be created before verification; a
///   best-effort parent pre-check avoids that in the common case, and a
///   failed verification rolls back our own creation. Worst case from a
///   lost parent race is an empty file, never foreign content destroyed.
/// - Parent-directory replacement after verification cannot redirect the
///   handle: writes address the verified object, whatever the name now
///   points at.
pub fn secure_open_write(
    path: &Path,
    sandbox_root: Option<&Path>,
    mode: WriteOpenMode,
) -> Result<SecureWriteFile, SecureOpenError> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        return secure_open_write_anchored(path, sandbox_root, mode);
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        return secure_open_write_unanchored(path, sandbox_root, mode);
    }
}

/// Unix anchored open: every parent component is opened relative to its
/// verified parent fd (`openat`), verified beneath the root, created with
/// `mkdirat` when missing (create mode), and the target itself is opened
/// with `openat` relative to the verified parent fd. No pathname in this
/// chain is ever trusted across a step boundary: each object is verified
/// before the next step proceeds, so a swapped component can only ever
/// deny the operation, never redirect it.
///
/// `root=None` (AllDirs scope) skips verification but keeps the same
/// anchored mechanics (plain `create_dir_all` + absolute open).
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn secure_open_write_anchored(
    path: &Path,
    sandbox_root: Option<&Path>,
    mode: WriteOpenMode,
) -> Result<SecureWriteFile, SecureOpenError> {
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::io::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

    // Anchor relative paths at the cwd and lexically normalize `..`
    // (callers pass absolute paths; this keeps the primitive total and
    // preserves path_allowed's canonicalizing behavior for legitimate
    // in-tree `a/../b` forms).
    let absolute: PathBuf = {
        let joined = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(path)
        };
        crate::agent_io::lexical_clean(&joined)
    };
    let file_name: std::ffi::OsString = absolute
        .file_name()
        .ok_or_else(|| {
            SecureOpenError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "no file name",
            ))
        })?
        .to_os_string();
    let parent = absolute.parent().unwrap_or_else(|| Path::new("/"));

    // Open + verify the sandbox root itself (AllDirs: no anchor needed).
    let mut cur: Option<OwnedFd> = None;
    if let Some(root) = sandbox_root {
        let root_file = File::open(root).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                SecureOpenError::NotFound
            } else {
                SecureOpenError::Io(e)
            }
        })?;
        verify_handle_object(&root_file, Some(root), root)?;
        // The walk below requires parent == root or beneath it lexically;
        // anything else was already denied upstream, but re-check here so
        // this primitive is safe standalone.
        if parent != root && !parent.starts_with(root) {
            return Err(SecureOpenError::SandboxDenied(format!(
                "Safefolder blocked path '{}' (parent outside current dir).",
                path.display()
            )));
        }
        // SAFETY: root_file is open; into_raw_fd transfers ownership once.
        cur = Some(unsafe { OwnedFd::from_raw_fd(root_file.into_raw_fd()) });
    }

    // Walk each parent component relative to the verified parent fd.
    if let Some(root) = sandbox_root {
        let rel = parent
            .strip_prefix(root)
            .map_err(|_| {
                SecureOpenError::SandboxDenied(format!(
                    "Safefolder blocked path '{}' (parent outside current dir).",
                    path.display()
                ))
            })?
            .to_path_buf();
        for component in rel.components() {
            let name = match component {
                std::path::Component::Normal(n) => n,
                // Lexically normalized inputs never contain these, but a
                // defensive skip beats trusting the caller.
                _ => continue,
            };
            let name_c = std::ffi::CString::new(name.as_bytes()).map_err(|_| {
                SecureOpenError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "bad path component",
                ))
            })?;
            let dirfd = cur
                .as_ref()
                .map(|f| f.as_raw_fd())
                .unwrap_or(libc::AT_FDCWD);
            // Retry loop: a racing mkdir (EEXIST) re-opens instead of failing.
            loop {
                // SAFETY: dirfd valid, name valid C string, no O_CREAT here.
                let fd = unsafe {
                    libc::openat(dirfd, name_c.as_ptr(), libc::O_RDONLY | libc::O_DIRECTORY)
                };
                if fd >= 0 {
                    // SAFETY: openat returned a fresh owned fd.
                    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
                    let f = File::from(owned);
                    let meta = std::fs::File::metadata(&f).map_err(SecureOpenError::Io)?;
                    if !meta.is_dir() {
                        return Err(SecureOpenError::Io(std::io::Error::new(
                            std::io::ErrorKind::InvalidInput,
                            "path component is not a directory",
                        )));
                    }
                    verify_handle_object(&f, Some(root), &parent.join(name))?;
                    // Ownership moves back fd-side (no close in between).
                    cur = Some(OwnedFd::from(f));
                    break;
                }
                let err = std::io::Error::last_os_error();
                let missing = err.kind() == std::io::ErrorKind::NotFound;
                if missing && matches!(mode, WriteOpenMode::OpenOrCreate) {
                    // SAFETY: dirfd valid, name valid, mode 0777 (umask applies,
                    // matching create_dir_all semantics).
                    let rc =
                        unsafe { libc::mkdirat(dirfd, name_c.as_ptr(), 0o777 as libc::mode_t) };
                    if rc == 0 {
                        continue; // created → open it on the next iteration
                    }
                    let mk_err = std::io::Error::last_os_error();
                    if mk_err.kind() == std::io::ErrorKind::AlreadyExists {
                        continue; // raced creation → open it
                    }
                    return Err(SecureOpenError::MkdirFailed(mk_err.to_string()));
                }
                if missing {
                    return Err(SecureOpenError::NotFound);
                }
                return Err(SecureOpenError::Io(err));
            }
        }
    } else if matches!(mode, WriteOpenMode::OpenOrCreate) {
        // AllDirs: no sandbox, plain recursive creation (parity with
        // create_dir_all).
        if let Err(e) = std::fs::create_dir_all(parent) {
            return Err(SecureOpenError::MkdirFailed(e.to_string()));
        }
    }

    // Open the target relative to the verified parent (or absolutely when
    // no anchor exists). No O_TRUNC: truncation before verification would
    // mutate an unverified object.
    let name_c = std::ffi::CString::new(file_name.as_bytes()).map_err(|_| {
        SecureOpenError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "bad file name",
        ))
    })?;
    // Best-effort pre-existence probe for creation rollback (lstat-style:
    // never follows the final symlink).
    let existed_before = match cur.as_ref() {
        Some(fd) => {
            let mut st: libc::stat = unsafe { std::mem::zeroed() };
            // SAFETY: dirfd valid, name valid, stat out-pointer valid.
            (unsafe {
                libc::fstatat(
                    fd.as_raw_fd(),
                    name_c.as_ptr(),
                    &mut st,
                    libc::AT_SYMLINK_NOFOLLOW,
                )
            } == 0)
        }
        None => path.symlink_metadata().is_ok(),
    };
    let mut flags = libc::O_RDWR;
    if matches!(mode, WriteOpenMode::OpenOrCreate) {
        flags |= libc::O_CREAT;
    }
    // SAFETY: dirfd (or AT_FDCWD) valid, name valid; mode only read with O_CREAT.
    let fd = unsafe {
        libc::openat(
            cur.as_ref()
                .map(|f| f.as_raw_fd())
                .unwrap_or(libc::AT_FDCWD),
            name_c.as_ptr(),
            flags,
            0o666 as libc::mode_t,
        )
    };
    if fd < 0 {
        let e = std::io::Error::last_os_error();
        if e.kind() == std::io::ErrorKind::NotFound {
            return Err(SecureOpenError::NotFound);
        }
        return Err(SecureOpenError::Io(e));
    }
    // SAFETY: openat returned a fresh owned fd exactly once.
    let owned = unsafe { OwnedFd::from_raw_fd(fd) };
    let file = File::from(owned);
    match verify_handle_object(&file, sandbox_root, path) {
        Ok(verified_path) => Ok(SecureWriteFile {
            file,
            verified_path,
            created_by_open: !existed_before,
        }),
        Err(e) => {
            // Roll back our own creation only: never delete a file that
            // already existed (it may belong to someone else). Removal is
            // fd-relative (`unlinkat` against the verified parent fd), so
            // even a namespace swap between verification failure and
            // cleanup cannot redirect the delete — the whole operation,
            // including failure cleanup, stays inside the FD-anchored
            // model instead of escaping back to pathname resolution.
            drop(file);
            if !existed_before {
                if let Some(parent_fd) = cur.as_ref() {
                    // SAFETY: parent fd verified live above; name valid.
                    // No AT_REMOVEDIR: only files roll back here, never
                    // directories (a replaced-with-dir entry fails safe).
                    unsafe {
                        libc::unlinkat(parent_fd.as_raw_fd(), name_c.as_ptr(), 0);
                    }
                } else {
                    // No anchor (AllDirs scope): nothing to escape to, and
                    // no boundary to defend — best-effort pathname removal.
                    let _ = std::fs::remove_file(path);
                }
            }
            Err(e)
        }
    }
}

/// Windows/other half of `secure_open_write`: no openat-style anchored
/// creation. Record the parent components missing before creation;
/// create them; open + verify the object; on verification failure roll
/// back our file AND any directories we created (empty-dir removal only
/// — remove_dir cannot destroy content, so rollback is side-effect-safe).
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn secure_open_write_unanchored(
    path: &Path,
    sandbox_root: Option<&Path>,
    mode: WriteOpenMode,
) -> Result<SecureWriteFile, SecureOpenError> {
    {
        let absolute: PathBuf = {
            let joined = if path.is_absolute() {
                path.to_path_buf()
            } else {
                std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(path)
            };
            crate::agent_io::lexical_clean(&joined)
        };
        let mut created_dirs: Vec<PathBuf> = Vec::new();
        if matches!(mode, WriteOpenMode::OpenOrCreate) {
            if let Some(parent) = absolute.parent() {
                // Lexically collect the missing chain (deepest first).
                let mut chain: Vec<PathBuf> = Vec::new();
                let mut probe = parent.to_path_buf();
                while probe.symlink_metadata().is_err() {
                    chain.push(probe.clone());
                    match probe.parent() {
                        Some(par) if par != probe => probe = par.to_path_buf(),
                        _ => break,
                    }
                }
                if let Err(e) = std::fs::create_dir_all(parent) {
                    return Err(SecureOpenError::MkdirFailed(e.to_string()));
                }
                created_dirs = chain;
            }
        }
        let existed_before = absolute.symlink_metadata().is_ok();
        let file = open_write_handle(&absolute, mode)?;
        match verify_handle_object(&file, sandbox_root, &absolute) {
            Ok(verified_path) => Ok(SecureWriteFile {
                file,
                verified_path,
                created_by_open: !existed_before,
            }),
            Err(e) => {
                drop(file);
                if !existed_before {
                    let _ = std::fs::remove_file(&absolute);
                }
                // Newest-first empty-dir rollback of our own creations.
                created_dirs.reverse();
                for dir in created_dirs {
                    let _ = std::fs::remove_dir(&dir);
                }
                Err(e)
            }
        }
    }
}

/// Platform write-handle open WITHOUT truncation (truncation before
/// verification would mutate an unverified object). Used only where no
/// anchored openat path exists (non-Unix); Unix goes through the
/// anchored walk above.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn open_write_handle(path: &Path, mode: WriteOpenMode) -> Result<File, SecureOpenError> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true).write(true);
        if matches!(mode, WriteOpenMode::OpenOrCreate) {
            opts.create(true);
        }
        opts.open(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                SecureOpenError::NotFound
            } else {
                SecureOpenError::Io(e)
            }
        })
    }
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use std::os::windows::io::{FromRawHandle, OwnedHandle};
        use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::Storage::FileSystem::{
            CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_DELETE, FILE_SHARE_READ,
            FILE_SHARE_WRITE, GENERIC_READ, GENERIC_WRITE, OPEN_ALWAYS, OPEN_EXISTING,
        };
        const ERROR_FILE_NOT_FOUND: u32 = 2;
        const ERROR_PATH_NOT_FOUND: u32 = 3;
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        // SAFETY: `wide` is NUL-terminated; other args are valid constants
        // or null; handle checked before use. No truncation here.
        let handle: HANDLE = unsafe {
            CreateFileW(
                wide.as_ptr(),
                (GENERIC_READ | GENERIC_WRITE) as u32,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                std::ptr::null(),
                if matches!(mode, WriteOpenMode::OpenOrCreate) {
                    OPEN_ALWAYS
                } else {
                    OPEN_EXISTING
                },
                FILE_ATTRIBUTE_NORMAL,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            // SAFETY: just failed; GetLastError is valid here.
            let code = unsafe { windows_sys::Win32::Foundation::GetLastError() };
            if code == ERROR_FILE_NOT_FOUND || code == ERROR_PATH_NOT_FOUND {
                return Err(SecureOpenError::NotFound);
            }
            return Err(SecureOpenError::Io(std::io::Error::from_raw_os_error(
                code as i32,
            )));
        }
        // SAFETY: valid owned handle exactly once.
        let owned = unsafe { OwnedHandle::from_raw_handle(handle as *mut std::ffi::c_void) };
        Ok(File::from(owned))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        let mut opts = std::fs::OpenOptions::new();
        opts.read(true).write(true);
        if matches!(mode, WriteOpenMode::OpenOrCreate) {
            opts.create(true);
        }
        opts.open(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                SecureOpenError::NotFound
            } else {
                SecureOpenError::Io(e)
            }
        })
    }
}
/// Resolve an open Windows HANDLE to its symlink-free `\\?\` object path.
/// Shared by the read and write verification paths (neither closes it).
#[cfg(windows)]
fn final_path_of_handle(
    handle: windows_sys::Win32::Foundation::HANDLE,
) -> Result<PathBuf, SecureOpenError> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_NAME_NORMALIZED, GetFinalPathNameByHandleW,
    };
    // SAFETY: handle is valid; null buffer with length 0 is the documented
    // length-query call.
    let needed =
        unsafe { GetFinalPathNameByHandleW(handle, std::ptr::null_mut(), 0, FILE_NAME_NORMALIZED) };
    if needed == 0 {
        return Err(SecureOpenError::Io(std::io::Error::last_os_error()));
    }
    let mut buf: Vec<u16> = vec![0; needed as usize + 1];
    // SAFETY: buf has the queried capacity; handle is valid.
    let written = unsafe {
        GetFinalPathNameByHandleW(
            handle,
            buf.as_mut_ptr(),
            buf.len() as u32,
            FILE_NAME_NORMALIZED,
        )
    };
    if written == 0 || written as usize >= buf.len() {
        return Err(SecureOpenError::Io(std::io::Error::last_os_error()));
    }
    buf.truncate(written as usize);
    Ok(PathBuf::from(std::ffi::OsString::from_wide(&buf)))
}
/// canonicalize → open → canonicalize again; the resolutions must match
/// and lie beneath the root. Narrows but does not fully close the window
/// (documented above).
#[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
pub fn secure_open_read_fallback(
    path: &Path,
    sandbox_root: Option<&Path>,
) -> Result<SecureFile, SecureOpenError> {
    let before = path.canonicalize().map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            SecureOpenError::NotFound
        } else {
            SecureOpenError::Io(e)
        }
    })?;
    if let Some(root) = sandbox_root {
        if !before.starts_with(root) {
            return Err(SecureOpenError::SandboxDenied(format!(
                "Safefolder blocked path '{}'.",
                path.display()
            )));
        }
    }
    let file = File::open(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            SecureOpenError::NotFound
        } else {
            SecureOpenError::Io(e)
        }
    })?;
    let after = path
        .canonicalize()
        .map_err(|_| SecureOpenError::SwappedDuringOpen)?;
    if after != before {
        return Err(SecureOpenError::SwappedDuringOpen);
    }
    Ok(SecureFile {
        verified_path: after,
        file,
    })
}

/// Read at most `max_bytes + 1` from an open handle (the +1 detects a
/// racing grow past the cap without trusting pre-read metadata).
pub fn read_capped(file: &mut File, max_bytes: u64) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::new();
    file.take(max_bytes + 1).read_to_end(&mut buf)?;
    Ok(buf)
}

/// One directory entry from an anchored listing: name, is-dir, size.
#[derive(Debug, Clone)]
pub struct AnchoredEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
}

/// List a directory anchored to a verified directory OBJECT (Unix):
/// open dir → verify beneath root via the handle → `fdopendir` → per-entry
/// `fstatat` relative to the directory fd. A parent/symlink swap after
/// verification cannot redirect the enumeration: every name resolves
/// against the verified dirfd, and vanished entries are skipped, never
/// misattributed.
#[cfg(unix)]
pub fn list_dir_anchored(
    dir: &Path,
    sandbox_root: Option<&Path>,
) -> Result<Vec<AnchoredEntry>, SecureOpenError> {
    use std::os::unix::io::AsRawFd;

    let dirf = File::open(dir).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            SecureOpenError::NotFound
        } else {
            SecureOpenError::Io(e)
        }
    })?;
    // Must actually be a directory (open succeeds on files too).
    let is_dir = std::fs::File::metadata(&dirf)
        .map(|m| m.is_dir())
        .unwrap_or(false);
    if !is_dir {
        return Err(SecureOpenError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a directory",
        )));
    }
    verify_handle_object(&dirf, sandbox_root, dir)?;

    // SAFETY: dirf is an O_RDONLY directory fd; fdopendir takes ownership
    // on success (we mem::forget dirf exactly then), closedir releases.
    let dirp = unsafe { libc::fdopendir(dirf.as_raw_fd()) };
    if dirp.is_null() {
        return Err(SecureOpenError::Io(std::io::Error::last_os_error()));
    }
    std::mem::forget(dirf);
    // SAFETY: dirfd() on our open DIR* is valid until closedir below.
    let dirfd = unsafe { libc::dirfd(dirp) };

    let mut out = Vec::new();
    loop {
        // SAFETY: readdir is thread-safe on glibc (per-DIR buffer); the
        // returned pointer is valid until the next readdir/closedir, and
        // we copy the name immediately. Null ends the stream.
        let entry = unsafe { libc::readdir(dirp) };
        if entry.is_null() {
            break;
        }
        // SAFETY: d_name is a NUL-terminated C string within entry.
        let name_c = unsafe { std::ffi::CStr::from_ptr((*entry).d_name.as_ptr()) };
        let name = name_c.to_string_lossy().into_owned();
        if name == "." || name == ".." {
            continue;
        }
        let name_cstr = match std::ffi::CString::new(name.clone()) {
            Ok(c) => c,
            Err(_) => continue, // interior NUL — skip, never serve
        };
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: dirfd valid, name valid C string, stat out-pointer valid.
        // No AT_SYMLINK_NOFOLLOW: matches File::metadata follow-behavior.
        let ok = unsafe { libc::fstatat(dirfd, name_cstr.as_ptr(), &mut stat, 0) } == 0;
        let (is_dir, size) = if ok {
            (
                (stat.st_mode as u32 & libc::S_IFMT as u32) == libc::S_IFDIR as u32,
                stat.st_size.max(0) as u64,
            )
        } else {
            (false, 0) // vanished mid-listing: keep name, zero size (parity)
        };
        out.push(AnchoredEntry { name, is_dir, size });
    }
    // SAFETY: closes the DIR* and its fd exactly once.
    unsafe { libc::closedir(dirp) };
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cwd_root() -> PathBuf {
        std::env::current_dir()
            .unwrap()
            .canonicalize()
            .unwrap_or_else(|_| std::env::current_dir().unwrap())
    }

    fn test_dir(tag: &str) -> PathBuf {
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let dir = std::env::current_dir().unwrap().join(format!(
            "target/hercules-securefs-test-{tag}-{n}-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Best-effort symlink creation for tests: Windows requires
    /// privilege/Developer Mode for symlinks; a refusal skips the test
    /// instead of failing it (the secure-open assertions below still run
    /// wherever links can be created).
    #[cfg(windows)]
    fn try_symlink_file(target: &Path, link: &Path) -> bool {
        std::os::windows::fs::symlink_file(target, link).is_ok()
    }

    #[test]
    fn static_out_of_tree_symlink_denied() {
        let dir = test_dir("static");
        let outside_dir =
            std::env::temp_dir().join(format!("hercules-securefs-outside-{}", std::process::id()));
        std::fs::create_dir_all(&outside_dir).unwrap();
        let secret = outside_dir.join("secret.txt");
        std::fs::write(&secret, "SECRET\n").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&secret, dir.join("link.txt")).unwrap();
        #[cfg(windows)]
        if !try_symlink_file(&secret, &dir.join("link.txt")) {
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_dir_all(&outside_dir);
            return;
        }

        #[cfg(not(any(unix, windows)))]
        {
            // No symlink primitive exercised on this platform.
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_dir_all(&outside_dir);
            return;
        }
        {
            let err = secure_open_read(&dir.join("link.txt"), Some(&cwd_root()))
                .expect_err("out-of-tree object must be denied");
            assert!(
                matches!(err, SecureOpenError::SandboxDenied(_)),
                "got: {err:?}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside_dir);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn open_handle_bound_despite_post_open_swap() {
        // Deterministic fd-binding proof: open while in-tree, THEN swap the
        // link out-of-tree; the handle must still yield in-tree bytes.
        let dir = test_dir("swap");
        let real = dir.join("real.txt");
        std::fs::write(&real, "IN TREE\n").unwrap();
        let outside_dir =
            std::env::temp_dir().join(format!("hercules-securefs-swapout-{}", std::process::id()));
        std::fs::create_dir_all(&outside_dir).unwrap();
        let secret = outside_dir.join("secret.txt");
        std::fs::write(&secret, "SECRET\n").unwrap();
        let link = dir.join("link.txt");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let mut sec = secure_open_read(&link, Some(&cwd_root())).expect("in-tree open allowed");
        // Swap AFTER open: handle still refers to the verified object.
        std::fs::remove_file(&link).unwrap();
        std::os::unix::fs::symlink(&secret, &link).unwrap();

        let bytes = read_capped(&mut sec.file, 1024).unwrap();
        assert_eq!(
            bytes, b"IN TREE\n",
            "fd-bound read must return the verified object"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside_dir);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn symlink_swap_hammer_never_escapes() {
        // The required regression test: hammer a symlink between an
        // in-tree and an out-of-tree file while reading through the
        // primitive. Out-of-tree bytes must NEVER be returned (either
        // in-tree bytes or an error — both are safe).
        use std::sync::atomic::{AtomicBool, Ordering};
        let dir = test_dir("hammer");
        let real = dir.join("real.txt");
        std::fs::write(&real, "IN TREE\n").unwrap();
        let outside_dir = std::env::temp_dir().join(format!(
            "hercules-securefs-hammerout-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&outside_dir).unwrap();
        let secret = outside_dir.join("secret.txt");
        std::fs::write(&secret, "SECRET\n").unwrap();
        let link = dir.join("link.txt");
        std::os::unix::fs::symlink(&real, &link).unwrap();

        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let stopper = stop.clone();
        let link_clone = link.clone();
        let real_clone = real.clone();
        let secret_clone = secret.clone();
        let swapper = std::thread::spawn(move || {
            let mut to_secret = true;
            while !stopper.load(Ordering::Relaxed) {
                let _ = std::fs::remove_file(&link_clone);
                let target = if to_secret {
                    &secret_clone
                } else {
                    &real_clone
                };
                let _ = std::os::unix::fs::symlink(target, &link_clone);
                to_secret = !to_secret;
            }
        });

        let root = cwd_root();
        let mut in_tree = 0u32;
        let mut denied = 0u32;
        for _ in 0..2000 {
            match secure_open_read(&link, Some(&root)) {
                Ok(mut sec) => {
                    if let Ok(bytes) = read_capped(&mut sec.file, 1024) {
                        assert_ne!(
                            bytes, b"SECRET\n",
                            "sandbox escape: out-of-tree bytes returned"
                        );
                        if bytes == b"IN TREE\n" {
                            in_tree += 1;
                        }
                    } else {
                        denied += 1;
                    }
                }
                Err(_) => denied += 1,
            }
        }
        stop.store(true, Ordering::Relaxed);
        swapper.join().unwrap();
        assert!(in_tree > 0, "hammer must include successful in-tree reads");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside_dir);
    }

    #[test]
    #[cfg(windows)]
    fn windows_in_tree_file_succeeds() {
        // Required case 1: a normal in-tree file opens and reads.
        let dir = test_dir("winnormal");
        let real = dir.join("real.txt");
        std::fs::write(&real, "IN TREE\n").unwrap();
        let mut sec = secure_open_read(&real, Some(&cwd_root())).expect("in-tree open allowed");
        let bytes = read_capped(&mut sec.file, 1024).unwrap();
        assert_eq!(bytes, b"IN TREE\n");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[cfg(windows)]
    fn windows_handle_bound_despite_post_open_swap() {
        // Required case 4: after a handle is opened, replacing the
        // directory entry cannot redirect the bytes read from that handle.
        let dir = test_dir("winswap");
        let real = dir.join("real.txt");
        std::fs::write(&real, "IN TREE\n").unwrap();
        let outside_dir = std::env::temp_dir().join(format!(
            "hercules-securefs-winswapout-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&outside_dir).unwrap();
        let secret = outside_dir.join("secret.txt");
        std::fs::write(&secret, "SECRET\n").unwrap();
        let link = dir.join("link.txt");
        if !try_symlink_file(&real, &link) {
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_dir_all(&outside_dir);
            return;
        }

        let mut sec = secure_open_read(&link, Some(&cwd_root())).expect("in-tree open allowed");
        // Swap AFTER open: the handle still refers to the verified object.
        std::fs::remove_file(&link).unwrap();
        if !try_symlink_file(&secret, &link) {
            // Privilege lost mid-test: the handle assertion below still
            // validates binding to the original object.
        }

        let bytes = read_capped(&mut sec.file, 1024).unwrap();
        assert_eq!(
            bytes, b"IN TREE\n",
            "handle-bound read must return the verified object"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside_dir);
    }

    #[test]
    #[cfg(windows)]
    fn windows_swap_hammer_never_escapes() {
        // Required case 3: hammer a reparse point between an in-tree and
        // an out-of-tree file while reading. Out-of-tree bytes must NEVER
        // be returned (either in-tree bytes or an error — both safe).
        use std::sync::atomic::{AtomicBool, Ordering};
        let dir = test_dir("winhammer");
        let real = dir.join("real.txt");
        std::fs::write(&real, "IN TREE\n").unwrap();
        let outside_dir = std::env::temp_dir().join(format!(
            "hercules-securefs-winhammerout-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&outside_dir).unwrap();
        let secret = outside_dir.join("secret.txt");
        std::fs::write(&secret, "SECRET\n").unwrap();
        let link = dir.join("link.txt");
        if !try_symlink_file(&real, &link) {
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_dir_all(&outside_dir);
            return;
        }

        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let stopper = stop.clone();
        let link_clone = link.clone();
        let real_clone = real.clone();
        let secret_clone = secret.clone();
        let swapper = std::thread::spawn(move || {
            let mut to_secret = true;
            while !stopper.load(Ordering::Relaxed) {
                let _ = std::fs::remove_file(&link_clone);
                let target = if to_secret {
                    &secret_clone
                } else {
                    &real_clone
                };
                if std::os::windows::fs::symlink_file(target, &link_clone).is_err() {
                    return;
                }
                to_secret = !to_secret;
            }
        });

        let root = cwd_root();
        let mut in_tree = 0u32;
        for _ in 0..2000 {
            match secure_open_read(&link, Some(&root)) {
                Ok(mut sec) => {
                    if let Ok(bytes) = read_capped(&mut sec.file, 1024) {
                        assert_ne!(
                            bytes, b"SECRET\n",
                            "sandbox escape: out-of-tree bytes returned"
                        );
                        if bytes == b"IN TREE\n" {
                            in_tree += 1;
                        }
                    }
                }
                Err(_) => {}
            }
        }
        stop.store(true, Ordering::Relaxed);
        swapper.join().unwrap();
        assert!(in_tree > 0, "hammer must include successful in-tree reads");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside_dir);
    }
}
