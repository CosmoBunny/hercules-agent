//! Lightweight hardware capabilities for backend selection.
//!
//! Exposes capability flags (can this host do CUDA/Metal/Vulkan/CPU,
//! unified memory?) instead of fragile GPU-name string matching. Detection
//! is conservative: unknown means false, and selection treats unknown as
//! CPU-only.

/// Host capabilities relevant to backend choice.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub os: String,
    pub arch: String,
    pub system_memory_bytes: u64,
    pub supports_cuda: bool,
    pub supports_metal: bool,
    pub supports_vulkan: bool,
    pub supports_cpu: bool,
    pub supports_unified_memory: bool,
}

impl HardwareInfo {
    /// Cheap, synchronous detection. Never probes daemons or loads models.
    pub fn detect() -> Self {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        let is_macos = os == "macos";
        let is_arm = arch == "aarch64";
        // Apple Silicon <=> macOS + ARM. Metal/unified memory follow it.
        let apple_silicon = is_macos && is_arm;
        // CUDA: NVIDIA device present (sysinfo-free heuristic via obvious
        // device nodes; conservative — absence of proof means false).
        let supports_cuda =
            cfg!(target_os = "linux") && std::path::Path::new("/dev/nvidia0").exists();
        // Vulkan: loader present is necessary but weak; still better than
        // GPU-name matching, and only used as a hint, never a gate.
        let supports_vulkan = std::path::Path::new("/usr/share/vulkan").exists()
            || std::path::Path::new("/usr/local/share/vulkan").exists()
            || apple_silicon
            || cfg!(target_os = "windows");
        Self {
            os,
            arch,
            system_memory_bytes: Self::system_memory(),
            supports_cuda,
            supports_metal: apple_silicon,
            supports_vulkan,
            supports_cpu: true,
            supports_unified_memory: apple_silicon,
        }
    }

    fn system_memory() -> u64 {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|t| {
                    t.lines()
                        .find(|l| l.starts_with("MemTotal:"))
                        .and_then(|l| {
                            l.split_whitespace()
                                .nth(1)
                                .and_then(|kb| kb.parse::<u64>().ok())
                                .map(|kb| kb * 1024)
                        })
                })
                .unwrap_or(0)
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }

    /// Short human summary for diagnostics/UI.
    pub fn summary(&self) -> String {
        let mut caps = vec!["CPU"];
        if self.supports_cuda {
            caps.push("CUDA");
        }
        if self.supports_metal {
            caps.push("Metal");
        }
        if self.supports_vulkan {
            caps.push("Vulkan");
        }
        if self.supports_unified_memory {
            caps.push("unified-mem");
        }
        format!(
            "{}-{}/{} ({})",
            self.os,
            self.arch,
            self.system_memory_bytes,
            caps.join("+")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_detect_sane() {
        let hw = HardwareInfo::detect();
        assert!(hw.supports_cpu);
        assert!(!hw.os.is_empty());
        // This CI host is Linux/x86_64: Metal must be off.
        if hw.os == "linux" && hw.arch == "x86_64" {
            assert!(!hw.supports_metal);
            assert!(!hw.supports_unified_memory);
        }
        assert!(!hw.summary().is_empty());
    }
}
