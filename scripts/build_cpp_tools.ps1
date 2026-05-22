param(
  [string]$RepoRoot = (Resolve-Path ".").Path,
  [string]$Configuration = "Release",
  [string]$GurobiHome = "",
  [ValidateSet("x64", "Win32", "ARM64", "ARM64EC")]
  [string]$Architecture = "x64",
  [int]$Parallel = 0,
  [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Invoke-NativeCommand {
  param(
    [string]$Command,
    [string[]]$Arguments
  )
  & $Command @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "$Command failed with exit code $LASTEXITCODE"
  }
}

function Find-BuiltExe {
  param(
    [string]$BuildDir,
    [string]$Name
  )
  $match = Get-ChildItem -Path $BuildDir -Recurse -Filter $Name -ErrorAction SilentlyContinue |
    Sort-Object FullName |
    Select-Object -First 1
  if (-not $match) {
    throw "Could not find built executable '$Name' under $BuildDir"
  }
  return $match.FullName
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
  throw "cmake not found. Run scripts\setup_cpp_windows.ps1 -InstallCMake first, then open a new PowerShell window."
}

if ([string]::IsNullOrWhiteSpace($GurobiHome)) {
  if ($env:GUROBI_HOME) {
    $GurobiHome = $env:GUROBI_HOME
  } else {
    $candidate = Get-ChildItem -Path "C:\" -Directory -Filter "gurobi*" -ErrorAction SilentlyContinue |
      ForEach-Object { Join-Path $_.FullName "win64" } |
      Where-Object { Test-Path $_ } |
      Sort-Object -Descending |
      Select-Object -First 1
    if ($candidate) {
      $GurobiHome = $candidate
    }
  }
}

if ([string]::IsNullOrWhiteSpace($GurobiHome) -or -not (Test-Path $GurobiHome)) {
  throw "Gurobi home not found. Pass -GurobiHome C:\gurobi1301\win64 or set GUROBI_HOME."
}

$buildRoot = Join-Path $RepoRoot "build"
$vnsBuild = Join-Path $buildRoot "VNS_cpp"
$bpcBuild = Join-Path $buildRoot "BPC_label_cpp"

if ($Clean) {
  foreach ($dir in @($vnsBuild, $bpcBuild)) {
    if (Test-Path $dir) {
      Write-Host "Removing stale build directory: $dir" -ForegroundColor Yellow
      Remove-Item $dir -Recurse -Force
    }
  }
}

$parallelArgs = @()
if ($Parallel -gt 0) {
  $parallelArgs = @("--parallel", "$Parallel")
} else {
  $parallelArgs = @("--parallel")
}

Write-Host "Building VNS_cpp..." -ForegroundColor Cyan
Invoke-NativeCommand "cmake" @(
  "-S", (Join-Path $RepoRoot "VNS_cpp"),
  "-B", $vnsBuild,
  "-G", "Visual Studio 17 2022",
  "-A", $Architecture
)
Invoke-NativeCommand "cmake" (@("--build", $vnsBuild, "--config", $Configuration) + $parallelArgs)
$vnsExe = Find-BuiltExe -BuildDir $vnsBuild -Name "vns_solver.exe"
Copy-Item $vnsExe (Join-Path $RepoRoot "VNS_cpp\vns_solver.exe") -Force

Write-Host "Building BPC_label_cpp..." -ForegroundColor Cyan
Invoke-NativeCommand "cmake" @(
  "-S", (Join-Path $RepoRoot "BPC_label_cpp"),
  "-B", $bpcBuild,
  "-G", "Visual Studio 17 2022",
  "-A", $Architecture,
  "-DGUROBI_HOME=$GurobiHome"
)
Invoke-NativeCommand "cmake" (@("--build", $bpcBuild, "--config", $Configuration) + $parallelArgs)
$bpcExe = Find-BuiltExe -BuildDir $bpcBuild -Name "bpc_label_solver.exe"
Copy-Item $bpcExe (Join-Path $RepoRoot "BPC_label_cpp\bpc_label_solver.exe") -Force

Write-Host "Built executables:" -ForegroundColor Green
Write-Host "  $(Join-Path $RepoRoot "VNS_cpp\vns_solver.exe")"
Write-Host "  $(Join-Path $RepoRoot "BPC_label_cpp\bpc_label_solver.exe")"
