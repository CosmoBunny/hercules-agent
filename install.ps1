<#
.SYNOPSIS
  Hercules Agent one-line installer — Windows, user-local, no admin.

.EXAMPLE
  irm https://raw.githubusercontent.com/CosmoBunny/hercules-agent/main/install.ps1 | iex

  # GPU build (auto-detected by default; override with $env:HERCULES_FLAVOR):
  $env:HERCULES_FLAVOR = 'nvidia'; irm .../install.ps1 | iex

  # Or save it first for -Flavor / -Version parameters:
  irm https://raw.githubusercontent.com/CosmoBunny/hercules-agent/main/install.ps1 -OutFile install.ps1
  .\install.ps1 -Flavor amd -Version v0.1.0a

Installs under $env:LOCALAPPDATA\hercules-agent and adds its bin/ to the
*user* PATH. No administrator rights needed.
#>
[CmdletBinding()]
param(
  [ValidateSet('auto', 'normal', 'nvidia', 'amd')]
  [string]$Flavor = 'auto',
  [string]$Version = '',
  [string]$InstallDir = ''
)

$ErrorActionPreference = 'Stop'
$Repo = 'CosmoBunny/hercules-agent'

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
  $InstallDir = Join-Path $env:LOCALAPPDATA 'hercules-agent'
}
if ([string]::IsNullOrWhiteSpace($Version)) {
  if (-not [string]::IsNullOrWhiteSpace($env:HERCULES_VERSION)) { $Version = $env:HERCULES_VERSION }
}
if ($Flavor -eq 'auto' -and -not [string]::IsNullOrWhiteSpace($env:HERCULES_FLAVOR)) {
  $Flavor = $env:HERCULES_FLAVOR
}

# --- 1. Detect the system --------------------------------------------------
$ArchName = $env:PROCESSOR_ARCHITECTURE
if ($ArchName -ne 'AMD64') {
  throw "Unsupported CPU architecture: $ArchName (only 64-bit x86 Windows builds are published)"
}
$Platform = 'windows'; $Arch = 'x86_64'

# --- 2. Pick the GPU flavor ------------------------------------------------
if ($Flavor -eq 'auto') {
  if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $Flavor = 'nvidia'
  } else {
    $amd = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -match 'AMD|Radeon' }
    $Flavor = if ($amd) { 'amd' } else { 'normal' }
  }
}
Write-Host "Detected: $Platform / $Arch, flavor: $Flavor"

# --- 3. Resolve the release and find our asset -----------------------------
if ([string]::IsNullOrWhiteSpace($Version)) {
  $Release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/latest"
} else {
  $Release = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases/tags/$Version"
}
$Tag = $Release.tag_name
$Names = $Release.assets | ForEach-Object { $_.name }

$matchFlavor = if ($Flavor -eq 'normal') {
  $Names | Where-Object { $_ -match "^hercules-agent-[0-9][^-]*-$Platform-$Arch\.tar\.gz$" -and $_ -notmatch '-(amd|nvidia)-' }
} else {
  $Names | Where-Object { $_ -match "^hercules-agent-$Flavor-[0-9][^-]*-$Platform-$Arch\.tar\.gz$" }
}
if (-not $matchFlavor) {
  if ($Flavor -ne 'normal') {
    Write-Warning "No $Flavor build for $Platform/$Arch in $Tag, falling back to the standard build."
    $matchFlavor = $Names | Where-Object { $_ -match "^hercules-agent-[0-9][^-]*-$Platform-$Arch\.tar\.gz$" -and $_ -notmatch '-(amd|nvidia)-' }
  }
  if (-not $matchFlavor) {
    throw "No build for $Platform/$Arch in release $Tag. Available: $($Names -join ', ')"
  }
}
$Asset = @($matchFlavor)[0]
Write-Host "Installing $Asset ($Tag)"

# --- 4. Download + verify --------------------------------------------------
$Temp = Join-Path $env:TEMP ("hercules-install-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Temp | Out-Null
try {
  $BaseUrl = "https://github.com/$Repo/releases/download/$Tag"
  $Archive = Join-Path $Temp $Asset
  Invoke-WebRequest "$BaseUrl/$Asset" -OutFile $Archive
  $ChecksumName = "$Asset.sha256"
  if ($Names -contains $ChecksumName) {
    $ChecksumFile = Join-Path $Temp $ChecksumName
    Invoke-WebRequest "$BaseUrl/$ChecksumName" -OutFile $ChecksumFile
    $Expected = ((Get-Content $ChecksumFile -Raw) -split '\s+')[0].ToLower()
    $Actual = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLower()
    if ($Expected -ne $Actual) { throw "Checksum mismatch for $Asset" }
    Write-Host 'Checksum OK'
  } else {
    Write-Warning "No .sha256 published for $Asset, skipping verification."
  }

  # --- 5. Install user-local -------------------------------------------------
  if (Test-Path $InstallDir) { Remove-Item $InstallDir -Recurse -Force }
  New-Item -ItemType Directory -Path $InstallDir | Out-Null
  tar.exe -xzf $Archive -C $InstallDir
  $BundleTop = Get-ChildItem $InstallDir -Directory | Select-Object -First 1
  $Exe = Join-Path $BundleTop.FullName 'bin\hercules.exe'
  if (-not (Test-Path $Exe)) { throw "Archive layout unexpected: no bin\hercules.exe under $($BundleTop.FullName)" }
  $BinDir = Join-Path $BundleTop.FullName 'bin'

  $UserPath = [Environment]::GetEnvironmentVariable('Path', 'User')
  if (($UserPath -split ';') -notcontains $BinDir) {
    [Environment]::SetEnvironmentVariable('Path', "$UserPath;$BinDir", 'User')
    Write-Host "Added to your user PATH: $BinDir"
  }
  if (($env:Path -split ';') -notcontains $BinDir) { $env:Path += ";$BinDir" }

  Write-Host ''
  Write-Host "Installed to $Exe"
  & $Exe --version
  Write-Host ''
  Write-Host 'Run it with: hercules.exe (restart your terminal first if PATH was just updated)'
} finally {
  Remove-Item $Temp -Recurse -Force -ErrorAction SilentlyContinue
}
