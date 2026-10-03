#!/usr/bin/env bash
# Hercules Agent one-line installer — Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/CosmoBunny/hercules-agent/main/install.sh | bash
#
# GPU builds (auto-detected, override if wrong):
#   curl -fsSL .../install.sh | bash -s -- --nvidia
#   curl -fsSL .../install.sh | bash -s -- --amd
#   curl -fsSL .../install.sh | HERCULES_VERSION=v0.1.0a bash
#
# Installs user-local only: binaries go under ~/.local (override with
# --prefix or HERCULES_PREFIX). No sudo, no system directories.
set -euo pipefail

REPO="CosmoBunny/hercules-agent"
PREFIX="${HERCULES_PREFIX:-$HOME/.local}"
FLAVOR="${HERCULES_FLAVOR:-auto}"
TAG="${HERCULES_VERSION:-latest}"

usage() {
  cat <<EOF
Usage: install.sh [--nvidia] [--amd] [--cpu] [--version TAG] [--prefix DIR]

  --nvidia / --amd / --cpu   GPU build flavor (default: auto-detect)
  --version TAG              install a pinned release tag (default: latest)
  --prefix DIR               install root holding bin/ and share/
                             (default: \$HOME/.local)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nvidia) FLAVOR="nvidia"; shift ;;
    --amd) FLAVOR="amd"; shift ;;
    --cpu|--normal) FLAVOR="normal"; shift ;;
    --version) TAG="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required tool: $1" >&2; exit 1; }; }
need curl; need tar

# --- 1. Detect the system -------------------------------------------------
OS="$(uname -s)"
case "$OS" in
  Linux) PLATFORM="linux" ;;
  Darwin) PLATFORM="macos" ;;
  *) echo "Unsupported OS: $OS (Linux and macOS only; Windows uses install.ps1)" >&2; exit 1 ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *) echo "Unsupported CPU architecture: $ARCH" >&2; exit 1 ;;
esac

# --- 2. Pick the GPU flavor ----------------------------------------------
if [[ "$FLAVOR" == "auto" ]]; then
  if [[ "$PLATFORM" == "macos" ]]; then
    FLAVOR="normal" # Metal acceleration is in the standard macOS build
  elif command -v nvidia-smi >/dev/null 2>&1; then
    FLAVOR="nvidia"
    echo "Reason: nvidia-smi is present"
  else
    AMD_LINE="$(lspci 2>/dev/null | grep -iE 'vga|3d controller|display controller' | grep -iE 'amd|radeon' | head -1 || true)"
    if [[ -n "$AMD_LINE" ]]; then
      FLAVOR="amd"
      echo "Reason: lspci reports AMD graphics: $AMD_LINE"
    else
      FLAVOR="normal"
      echo "Reason: no NVIDIA (no nvidia-smi) and no AMD display device in lspci"
    fi
  fi
elif [[ -n "${HERCULES_FLAVOR:-}" ]]; then
  echo "Reason: flavor forced by HERCULES_FLAVOR=$HERCULES_FLAVOR"
fi
case "$FLAVOR" in
  normal|cpu) FLAVOR="normal" ;;
  nvidia|amd) ;;
  *) echo "Unknown flavor: $FLAVOR (want: normal, nvidia, amd)" >&2; exit 1 ;;
esac

echo "Detected: $PLATFORM / $ARCH, flavor: $FLAVOR"

# --- 3. Resolve the release and find our asset ----------------------------
API="https://api.github.com/repos/$REPO/releases"
if [[ "$TAG" == "latest" ]]; then
  RELEASE_JSON="$(curl -fsSL "$API/latest")"
else
  RELEASE_JSON="$(curl -fsSL "$API/tags/$TAG")"
fi
TAG="$(printf '%s' "$RELEASE_JSON" | grep -o '"tag_name": *"[^"]*"' | head -1 | sed 's/.*": *"//;s/"//')"
ASSETS="$(printf '%s' "$RELEASE_JSON" | grep -o '"name": *"[^"]*"' | sed 's/.*": *"//;s/"//')"
[[ -n "$TAG" ]] || { echo "Could not resolve a release from GitHub API" >&2; exit 1; }

pick_asset() { # $1 = flavor infix ("" for normal, "-nvidia" etc.)
  printf '%s\n' "$ASSETS" | grep -E "^hercules-agent$1-[0-9][^-]*-${PLATFORM}-${ARCH}\\.tar\\.gz$" | head -1 || true
}

if [[ "$FLAVOR" == "normal" ]]; then
  # Plain build has no flavor infix; exclude the GPU ones explicitly.
  ASSET="$(printf '%s\n' "$ASSETS" | grep -E "^hercules-agent-[0-9][^-]*-${PLATFORM}-${ARCH}\\.tar\\.gz$" | grep -vE -- '-amd-|-nvidia-' | head -1 || true)"
else
  ASSET="$(pick_asset "-$FLAVOR")"
  if [[ -z "$ASSET" ]]; then
    echo "No $FLAVOR build for $PLATFORM/$ARCH in $TAG, falling back to the standard build." >&2
    ASSET="$(printf '%s\n' "$ASSETS" | grep -E "^hercules-agent-[0-9][^-]*-${PLATFORM}-${ARCH}\\.tar\\.gz$" | grep -vE -- '-amd-|-nvidia-' | head -1 || true)"
  fi
fi
[[ -n "$ASSET" ]] || {
  echo "No build for $PLATFORM/$ARCH in release $TAG." >&2
  echo "Available assets:" >&2
  printf '  %s\n' "$ASSETS" >&2
  exit 1
}

echo "Installing $ASSET ($TAG)"

# --- 4. Download + verify --------------------------------------------------
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
BASE_URL="https://github.com/$REPO/releases/download/$TAG"
curl -fsSL -o "$ASSET" "$BASE_URL/$ASSET"
if printf '%s\n' "$ASSETS" | grep -qxF "$ASSET.sha256"; then
  curl -fsSL -o "$ASSET.sha256" "$BASE_URL/$ASSET.sha256"
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$TMP" && sha256sum -c "$ASSET.sha256")
  else
    (cd "$TMP" && shasum -a 256 -c "$ASSET.sha256")
  fi
  echo "Checksum OK"
else
  echo "Warning: no .sha256 published for $ASSET, skipping verification." >&2
fi

# --- 5. Install user-local -------------------------------------------------
SHARE="$PREFIX/share/hercules-agent"
BIN_DIR="$PREFIX/bin"
rm -rf "$SHARE"
mkdir -p "$SHARE" "$BIN_DIR"
tar -xzf "$ASSET" -C "$SHARE"
# The bundle is one top-level dir holding bin/hercules + resources.
BUNDLE_TOP="$(find "$SHARE" -maxdepth 1 -mindepth 1 -type d | head -1)"
EXE="$BUNDLE_TOP/bin/hercules"
[[ -x "$EXE" ]] || { echo "Archive layout unexpected: no bin/hercules under $BUNDLE_TOP" >&2; exit 1; }
ln -sf "$EXE" "$BIN_DIR/hercules"

echo
echo "Installed to $EXE"
echo "Symlinked  $BIN_DIR/hercules"
"$BIN_DIR/hercules" --version || true
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo
    echo "$BIN_DIR is not on your PATH. Add it with:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac
echo
echo "Run it with: hercules"
