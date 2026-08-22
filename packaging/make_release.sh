#!/usr/bin/env bash
# =============================================================================
# make_release.sh
# Compile release CLI binary for Hercules Agent, bundle it, and copy the archive
# into the main project `release/` folder (for user downloads / CI releases).
#
# Usage:
#   ./packaging/make_release.sh
#   ./packaging/make_release.sh --no-build
#   ./packaging/make_release.sh --features "gpu"
#   ./packaging/make_release.sh --features "llama-cpp-static"
#   ./packaging/make_release.sh --no-default-features
#
# Output (example on Linux x86_64):
#   dist/hercules-agent-0.1.0-linux-x86_64/
#   dist/hercules-agent-0.1.0-linux-x86_64.tar.gz
#   release/hercules-agent-0.1.0-linux-x86_64.tar.gz
#   release/hercules-agent-0.1.0-linux-x86_64.tar.gz.sha256
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

VERSION="$(
  sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -1
)"
VERSION="${VERSION:-0.1.0}"

DO_BUILD=1
CARGO_EXTRA=()
NO_DEFAULT=0
BIN_NAME="hercules"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) DO_BUILD=0; shift ;;
    --bin) BIN_NAME="$2"; shift 2 ;;
    --no-default-features) NO_DEFAULT=1; shift ;;
    --features)
      CARGO_EXTRA+=(--features "$2"); shift 2 ;;
    -h|--help)
      sed -n '2,20p' "$0"; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH_TAG="x86_64" ;;
  aarch64|arm64) ARCH_TAG="aarch64" ;;
  *) ARCH_TAG="$ARCH" ;;
esac

case "$OS" in
  linux)  PLATFORM="linux" ;;
  darwin) PLATFORM="macos" ;;
  mingw*|msys*|cygwin*|windows*) PLATFORM="windows" ;;
  *) PLATFORM="$OS" ;;
esac

BUNDLE_NAME="hercules-agent-${VERSION}-${PLATFORM}-${ARCH_TAG}"
DIST_DIR="$ROOT/dist"
BUNDLE_DIR="$DIST_DIR/$BUNDLE_NAME"
ARCHIVE_TGZ="$DIST_DIR/${BUNDLE_NAME}.tar.gz"
ARCHIVE_ZIP="$DIST_DIR/${BUNDLE_NAME}.zip"
RELEASE_DIR="$ROOT/release"

EXE_NAME="${BIN_NAME}"
if [[ "$PLATFORM" == "windows" ]]; then
  EXE_NAME="${BIN_NAME}.exe"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Hercules Agent CLI Release Build"
echo " Version : $VERSION"
echo " Host    : $PLATFORM / $ARCH_TAG"
echo " Bundle  : $BUNDLE_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Build ────────────────────────────────────────────────────────────────────
if [[ $DO_BUILD -eq 1 ]]; then
  echo
  echo "▶ cargo build --release …"
  BUILD_ARGS=(build --release --bin "$BIN_NAME")
  if [[ $NO_DEFAULT -eq 1 ]]; then
    BUILD_ARGS+=(--no-default-features)
  fi
  if [[ ${#CARGO_EXTRA[@]} -gt 0 ]]; then
    BUILD_ARGS+=("${CARGO_EXTRA[@]}")
  fi
  cargo "${BUILD_ARGS[@]}"
else
  echo "▶ Skipping build (--no-build)"
fi

REL_BIN_DIR="$ROOT/target/release"
if [[ ! -f "$REL_BIN_DIR/$EXE_NAME" ]]; then
  echo "✗ Missing binary: $REL_BIN_DIR/$EXE_NAME" >&2
  echo "  Run without --no-build, or build it first." >&2
  exit 1
fi

# ── Stage bundle ─────────────────────────────────────────────────────────────
echo
echo "▶ Staging bundle at $BUNDLE_DIR"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/bin"

cp "$REL_BIN_DIR/$EXE_NAME" "$BUNDLE_DIR/bin/$EXE_NAME"
chmod +x "$BUNDLE_DIR/bin/$EXE_NAME" 2>/dev/null || true

for f in README.md LICENSE TODO.md; do
  if [[ -f "$ROOT/$f" ]]; then
    cp "$ROOT/$f" "$BUNDLE_DIR/$f"
  fi
done

cat > "$BUNDLE_DIR/README.txt" <<EOF
Hercules Agent ${VERSION} (${PLATFORM} ${ARCH_TAG})
===================================================

Hercules Agent — Local AI coding agent CLI and TUI.

Included Binary:
  bin/${EXE_NAME}

Usage:
  ./bin/${EXE_NAME}

Install:
  Copy 'bin/${EXE_NAME}' to a directory in your PATH (e.g. /usr/local/bin or ~/.local/bin).

For configuration options and backends (llama.cpp, Ollama), see README.md.
EOF

# ── Archive ──────────────────────────────────────────────────────────────────
echo "▶ Creating archive …"
mkdir -p "$DIST_DIR" "$RELEASE_DIR"
FINAL_ARCHIVE=""

if [[ "$PLATFORM" == "windows" ]] && command -v zip >/dev/null 2>&1; then
  rm -f "$ARCHIVE_ZIP"
  (cd "$DIST_DIR" && zip -r -q "$(basename "$ARCHIVE_ZIP")" "$BUNDLE_NAME")
  FINAL_ARCHIVE="$ARCHIVE_ZIP"
else
  tar -C "$DIST_DIR" -czf "$ARCHIVE_TGZ" "$BUNDLE_NAME"
  FINAL_ARCHIVE="$ARCHIVE_TGZ"
fi

if [[ -z "${FINAL_ARCHIVE}" || ! -f "$FINAL_ARCHIVE" ]]; then
  echo "✗ Failed to create archive" >&2
  exit 1
fi

# ── Copy into release/ ───────────────────────────────────────────────────────
cp -f "$FINAL_ARCHIVE" "$RELEASE_DIR/"
BASENAME="$(basename "$FINAL_ARCHIVE")"

# SHA256 for integrity
if command -v sha256sum >/dev/null 2>&1; then
  (cd "$RELEASE_DIR" && sha256sum "$BASENAME" > "${BASENAME}.sha256")
elif command -v shasum >/dev/null 2>&1; then
  (cd "$RELEASE_DIR" && shasum -a 256 "$BASENAME" > "${BASENAME}.sha256")
fi

# Stable latest link / copy
LATEST_EXT="${BASENAME##*.}"
if [[ "$BASENAME" == *.tar.gz ]]; then
  LATEST_EXT="tar.gz"
elif [[ "$BASENAME" == *.zip ]]; then
  LATEST_EXT="zip"
fi
LATEST_LINK="$RELEASE_DIR/hercules-agent-latest-${PLATFORM}-${ARCH_TAG}.${LATEST_EXT}"
rm -f "$RELEASE_DIR/hercules-agent-latest-${PLATFORM}-${ARCH_TAG}".* 2>/dev/null || true
ln -s "$BASENAME" "$LATEST_LINK" 2>/dev/null || cp -f "$RELEASE_DIR/$BASENAME" "$LATEST_LINK"

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Done"
echo " Bundle dir : $BUNDLE_DIR"
echo " Archive    : $FINAL_ARCHIVE"
echo " Download   : $RELEASE_DIR/$BASENAME"
echo " Latest     : $LATEST_LINK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
