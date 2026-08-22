#!/usr/bin/env bash
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
FLAVOR="normal"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) DO_BUILD=0; shift ;;
    --bin) BIN_NAME="$2"; shift 2 ;;
    --no-default-features) NO_DEFAULT=1; shift ;;
    --flavor) FLAVOR="$2"; shift 2 ;;
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

if [[ "$FLAVOR" != "normal" ]]; then
  BUNDLE_NAME="hercules-agent-${FLAVOR}-${VERSION}-${PLATFORM}-${ARCH_TAG}"
else
  BUNDLE_NAME="hercules-agent-${VERSION}-${PLATFORM}-${ARCH_TAG}"
fi

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
echo " Flavor  : $FLAVOR"
echo " Bundle  : $BUNDLE_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

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
  exit 1
fi

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

cat > "$BUNDLE_DIR/README.txt" <<README_EOF
Hercules Agent ${VERSION} (${PLATFORM} ${ARCH_TAG}) - Flavor: ${FLAVOR}
===================================================

Hercules Agent — Local AI coding agent CLI and TUI.

Included Binary:
  bin/${EXE_NAME}

Usage:
  ./bin/${EXE_NAME}

Install:
  Copy 'bin/${EXE_NAME}' to a directory in your PATH (e.g. /usr/local/bin or ~/.local/bin).
README_EOF

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

cp -f "$FINAL_ARCHIVE" "$RELEASE_DIR/"
BASENAME="$(basename "$FINAL_ARCHIVE")"

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$RELEASE_DIR" && sha256sum "$BASENAME" > "${BASENAME}.sha256")
elif command -v shasum >/dev/null 2>&1; then
  (cd "$RELEASE_DIR" && shasum -a 256 "$BASENAME" > "${BASENAME}.sha256")
fi

echo " Done"
echo " Archive    : $FINAL_ARCHIVE"
