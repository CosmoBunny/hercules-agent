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

# --- 2. Hardware info (also drives flavor auto-detect) --------------------
if [[ "$PLATFORM" == "macos" ]]; then
  CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "$ARCH")"
  GPU="Apple Silicon (Metal)"
  RAM="$(sysctl -n hw.memsize 2>/dev/null | awk '{printf "%.0fG", $1/1024/1024/1024}')"
else
  CPU="$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | xargs || echo "$ARCH")"
  CPU="$CPU ($(nproc 2>/dev/null || echo ?) cores)"
  GPU_LINES="$(lspci 2>/dev/null | grep -iE 'vga|3d controller|display controller' || true)"
  GPU="$(printf '%s' "$GPU_LINES" | head -1 | sed 's/^.*VGA compatible controller: //;s/^.*3D controller: //;s/^.*Display controller: //;s/ (rev.*//' | xargs || true)"
  [[ -n "$GPU" ]] || GPU="unknown"
  RAM="$(free -h 2>/dev/null | awk '/^Mem:/{print $2}')"
  RAM="${RAM%i}"
  [[ -n "$RAM" ]] || RAM="unknown"
fi

if [[ "$FLAVOR" == "auto" ]]; then
  if [[ "$PLATFORM" == "macos" ]]; then
    FLAVOR="normal" # Metal acceleration is in the standard macOS build
  elif command -v nvidia-smi >/dev/null 2>&1; then
    FLAVOR="nvidia"
  elif printf '%s' "$GPU_LINES" | grep -qiE 'amd|radeon'; then
    FLAVOR="amd"
  else
    FLAVOR="normal"
  fi
fi
case "$FLAVOR" in
  normal|cpu) FLAVOR="normal" ;;
  nvidia|amd) ;;
  *) echo "Unknown flavor: $FLAVOR (want: normal, nvidia, amd)" >&2; exit 1 ;;
esac

pretty_flavor() { case "$1" in
  normal) echo "Normal" ;; nvidia) echo "Nvidia" ;; amd) echo "Amd" ;;
esac; }
pretty_os() { case "$1" in
  linux) echo "Linux" ;; macos) echo "Mac" ;;
esac; }

# --- 3. Banner ------------------------------------------------------------
# Actual splash.txt, printed raw. When run from a repo checkout
# (./install.sh) the local file is used so new artwork shows
# immediately; the piped one-liner fetches it from GitHub instead.
LINE="-------------------------------------------------------------------------"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [[ -f "$SCRIPT_DIR/splash.txt" ]]; then
  SPLASH="$(cat "$SCRIPT_DIR/splash.txt")"
else
  SPLASH="$(curl -fsSL --max-time 15 "https://raw.githubusercontent.com/$REPO/main/splash.txt" 2>/dev/null || true)"
fi
if [[ -z "$SPLASH" ]]; then
  SPLASH="HERCULES AGENT"
fi
# Hardware info rides in a second column beside the splash — but only
# when the terminal is wide enough. Every art line is exactly 30
# columns, so plain concatenation aligns; narrow terminals get the
# stacked layout instead of wrapped garbage.
HWINFO_LINES=("CPU : $CPU" "GPU : $GPU" "RAM : $RAM")
COLS="${COLUMNS:-$(tput cols 2>/dev/null || echo 80)}"
HW_W=0
for h in "${HWINFO_LINES[@]}"; do
  (( ${#h} > HW_W )) && HW_W=${#h}
done
WIDE=1
(( COLS < 30 + 2 + HW_W + 2 )) && WIDE=0
i=0
while IFS= read -r sline; do
  if (( WIDE )) && (( i < ${#HWINFO_LINES[@]} )); then
    printf '%s  %s\n' "$sline" "${HWINFO_LINES[$i]}"
  else
    printf '%s\n' "$sline"
  fi
  i=$((i + 1))
done <<<"$SPLASH"
if (( ! WIDE )); then
  printf '%s\n' "${HWINFO_LINES[@]}"
fi
echo "$LINE"
echo "> Downloading Hercules | $(pretty_flavor "$FLAVOR") | $(pretty_os "$PLATFORM") ($ARCH)"

# --- 4. Resolve the release and find our asset ----------------------------
API="https://api.github.com/repos/$REPO/releases"
if [[ "$TAG" == "latest" ]]; then
  RELEASE_JSON="$(curl -fsSL "$API/latest")"
else
  RELEASE_JSON="$(curl -fsSL "$API/tags/$TAG")"
fi
TAG="$(printf '%s' "$RELEASE_JSON" | grep -o '"tag_name": *"[^"]*"' | head -1 | sed 's/.*": *"//;s/"//')"
ASSETS="$(printf '%s' "$RELEASE_JSON" | grep -o '"name": *"[^"]*"' | sed 's/.*": *"//;s/"//')"
[[ -n "$TAG" ]] || { echo "Could not resolve a release from GitHub API" >&2; exit 1; }

if [[ "$FLAVOR" == "normal" ]]; then
  # Plain build has no flavor infix; exclude the GPU ones explicitly.
  ASSET="$(printf '%s\n' "$ASSETS" | grep -E "^hercules-agent-[0-9][^-]*-${PLATFORM}-${ARCH}\\.tar\\.gz$" | grep -vE -- '-amd-|-nvidia-' | head -1 || true)"
else
  ASSET="$(printf '%s\n' "$ASSETS" | grep -E "^hercules-agent-${FLAVOR}-[0-9][^-]*-${PLATFORM}-${ARCH}\\.tar\\.gz$" | head -1 || true)"
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

echo "  Package: $ASSET ($TAG)"

# --- 5. Download + verify --------------------------------------------------
TMP="$(mktemp -d)"
trap 'tput cnorm 2>/dev/null || true; rm -rf "$TMP"' EXIT
cd "$TMP"
# Download with a terminal-width progress bar: ###--- (x/y MB) pct%.
# Sized from $COLS so it can never wrap into garbage; silent when
# output is not a tty.
download_with_progress() { # $1 = url, $2 = outfile
  local url="$1" out="$2"
  local total pid done_now pct bar_w fill rest bar
  total="$(curl -fsSLI --max-time 20 "$url" 2>/dev/null | grep -i '^content-length:' | tr -d '\r' | awk '{print $2}' | tail -1 || true)"
  # Pre-create so size polls never race curl's file creation (a
  # missing file makes the '<' redirect itself print an error that
  # 2>/dev/null cannot suppress, and each error newline breaks the
  # \r redraw into stacked duplicate bars).
  : > "$out"
  curl -fsSL --max-time 900 -o "$out" "$url" &
  pid=$!
  start_s=$SECONDS
  SLOW_ROASTS=(
    "Library WiFi and your network - no difference at all."
    "Feeling bad for your ISP. They can't provide that much speed."
    "My ice cream is melting. Please go fast."
    "Downloading at this speed? The model will finish training first."
    "Are you downloading via carrier pigeon?"
    "Dial-up called. It wants its speed back."
    "Is your internet powered by a hamster on a wheel?"
    "This speed makes dial-up look ambitious."
    "I've seen glacial ice melt faster than this download."
    "Your packets appear to be traveling by surface mail."
    "Even a carrier pigeon would request hazard pay for this route."
    "At this rate the file will arrive sometime next fiscal year."
  )
  MID_QUIPS=(
    "Respectable. Like instant noodles - gets the job done."
    "Not bad. Your ISP showed up to work today."
    "Steady. No awards, no complaints."
    "Mid speed, mid day, still downloading. It is what it is."
    "Solid performance. Neither spectacular nor disappointing."
    "Your ISP is meeting the contractual minimum today."
    "Consistent throughput. Reliable, if unremarkable."
    "Average velocity for an average afternoon."
    "Functional and serviceable. No further comment required."
  )
  FAST_PRAISE=(
    "Certified fiber enjoyer. Blink and you'll miss it."
    "Your ISP deserves a raise."
    "Ludicrous speed. Go plaid."
    "That download was faster than my last relationship."
    "Blazing throughput. Your neighbors are almost certainly jealous."
    "Fiber optic excellence operating at full capacity."
    "This connection could download the internet itself."
    "Your download just established a new personal best."
    "Speed that borders on the unreasonable. Impressive."
  )
  # Quote tier by speed (MB/s): roast / quip / praise.
  speed_tier() { # $1 = MB/s float -> echoes 0, 1 or 2
    awk -v m="$1" 'BEGIN { print (m < 1.0) ? 0 : ((m > 25.0) ? 2 : 1) }'
  }
  pick_for_tier() { # $1 = 0, 1 or 2 -> echoes one rotating line
    case "$1" in
      0) printf '%s' "${SLOW_ROASTS[$(( RANDOM % ${#SLOW_ROASTS[@]} ))]}" ;;
      2) printf '%s' "${FAST_PRAISE[$(( RANDOM % ${#FAST_PRAISE[@]} ))]}" ;;
      *) printf '%s' "${MID_QUIPS[$(( RANDOM % ${#MID_QUIPS[@]} ))]}" ;;
    esac
  }
  if [[ -t 2 ]] && [[ -n "$total" ]] && (( total > 0 )); then
    # Two live lines — bar on top, full quote below — redrawn with \r
    # plus one cursor-up. The cursor stays hidden throughout (civis),
    # so no jumping is ever visible. Both lines are padded to a fixed
    # width, so overwrites are total and no tails survive. Narrow/dumb
    # terminals fall back to the single-line pct + quote.
    bar_w=$(( COLS - 32 ))
    (( bar_w < 10 )) && bar_w=10
    LW=$(( COLS - 1 ))
    (( LW < 20 )) && LW=20
    TWO_LINE=1
    (( COLS < 78 )) && TWO_LINE=0
    [[ "${TERM:-}" == "dumb" ]] && TWO_LINE=0
    n=0
    first=1
    quip="$(pick_for_tier 1)"
    qcap=$(( COLS - 7 ))
    (( qcap < 10 )) && qcap=10
    tput civis 2>/dev/null || true # hide cursor for the live lines
    while kill -0 "$pid" 2>/dev/null; do
      if [[ -f "$out" ]]; then
        done_now="$(wc -c < "$out" 2>/dev/null || echo 0)"
      else
        done_now=0
      fi
      done_now="${done_now//[[:space:]]/}"
      pct=$(( done_now * 100 / total ))
      (( pct > 100 )) && pct=100
      n=$(( n + 1 ))
      # Fresh quote ~every 5s, tiered by our average speed so far.
      if (( n % 10 == 1 )); then
        sofar_s=$(( SECONDS - start_s ))
        (( sofar_s <= 0 )) && sofar_s=1
        sofar_mbps="$(awk -v b="$done_now" -v e="$sofar_s" 'BEGIN { printf "%.1f", (b / 1048576) / e }')"
        quip="$(pick_for_tier "$(speed_tier "$sofar_mbps")")"
      fi
      if (( TWO_LINE )); then
        fill=$(( pct * bar_w / 100 ))
        rest=$(( bar_w - fill ))
        bar="$(printf '%*s' "$fill" '' | tr ' ' '#')$(printf '%*s' "$rest" '' | tr ' ' '-')"
        pctstr="$(printf '%3s' "$pct")%%"
        (( first )) || printf '\033[1A' >&2
        first=0
        printf "\r%-${LW}s\n" "  $bar $pctstr" >&2
        printf "\r%-${LW}s" "  > $quip" >&2
      else
        printf '\r  %3s%% > %-*s' "$pct" "$qcap" "${quip:0:$qcap}" >&2
      fi
      sleep 0.5
    done
    printf '\n' >&2
    tput cnorm 2>/dev/null || true # cursor back
  fi
  wait "$pid" || { echo "Download failed: $url" >&2; exit 1; }
  # Network speed verdict: one rotating roast (or praise) per tier.
  elapsed=$(( SECONDS - start_s ))
  (( elapsed <= 0 )) && elapsed=1
  final_bytes="$(wc -c < "$out" 2>/dev/null || echo 0)"
  final_bytes="${final_bytes//[[:space:]]/}"
  mbps="$(awk -v b="$final_bytes" -v e="$elapsed" 'BEGIN { printf "%.1f", (b / 1048576) / e }')"
  if (( $(speed_tier "$mbps") == 0 )); then
    pick="$(pick_for_tier 0)"
  elif (( $(speed_tier "$mbps") == 2 )); then
    pick="$(pick_for_tier 2)"
  else
    pick="$(pick_for_tier 1)"
  fi
  echo "  Speed: ${mbps} MB/s - $pick"
}

BASE_URL="https://github.com/$REPO/releases/download/$TAG"
download_with_progress "$BASE_URL/$ASSET" "$ASSET"
echo "  Downloaded $ASSET ($(du -h "$ASSET" | cut -f1))"
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

# --- 6. Install user-local -------------------------------------------------
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

echo "$LINE"
echo "  Installed : $EXE"
echo "  Symlinked : $BIN_DIR/hercules"
"$BIN_DIR/hercules" --version || true
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    echo "  NOTE: $BIN_DIR is not on your PATH. Add it with:"
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    ;;
esac
echo "$LINE"
echo "Run it with: hercules"
