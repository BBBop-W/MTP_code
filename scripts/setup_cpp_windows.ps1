param(
  [switch]$InstallBuildTools,
  [switch]$InstallCMake,
  [switch]$InstallAll,
  [switch]$ManualInstructions
)

$ErrorActionPreference = "Stop"

function Write-ManualInstructions {
  Write-Host ""
  Write-Host "Manual install path for machines without winget:" -ForegroundColor Yellow
  Write-Host ""
  Write-Host "1. Install Visual Studio 2022 Build Tools." -ForegroundColor Cyan
  Write-Host "   Download: https://aka.ms/vs/17/release/vs_BuildTools.exe"
  Write-Host "   Silent command after download:"
  Write-Host "     .\vs_BuildTools.exe --quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  Write-Host ""
  Write-Host "2. Install CMake for Windows x64." -ForegroundColor Cyan
  Write-Host "   Download: https://cmake.org/download/"
  Write-Host "   During installation, choose to add CMake to PATH for all users."
  Write-Host ""
  Write-Host "3. Install Gurobi separately and activate its license." -ForegroundColor Cyan
  Write-Host "   Then set GUROBI_HOME to the win64 folder, for example:"
  Write-Host "     [Environment]::SetEnvironmentVariable('GUROBI_HOME', 'C:\gurobi1301\win64', 'Machine')"
  Write-Host "     `$env:Path += ';C:\gurobi1301\win64\bin'"
  Write-Host ""
  Write-Host "4. Open a new PowerShell window and build/run again." -ForegroundColor Cyan
  Write-Host "     .\scripts\build_cpp_tools.ps1 -GurobiHome C:\gurobi1301\win64"
}

if ($InstallAll) {
  $InstallBuildTools = $true
  $InstallCMake = $true
}

if ($ManualInstructions) {
  Write-ManualInstructions
  exit 0
}

if (-not $InstallBuildTools -and -not $InstallCMake) {
  Write-Host "Nothing selected. Use -InstallAll, -InstallBuildTools, -InstallCMake, or -ManualInstructions." -ForegroundColor Yellow
  exit 0
}

$winget = Get-Command winget -ErrorAction SilentlyContinue
$choco = Get-Command choco -ErrorAction SilentlyContinue

if (-not $winget -and -not $choco) {
  Write-Host "Neither winget nor Chocolatey was found, so this script cannot install packages automatically." -ForegroundColor Red
  Write-Host "If Microsoft Store is available, installing 'App Installer' will provide winget." -ForegroundColor Yellow
  Write-ManualInstructions
  exit 1
}

if ($InstallBuildTools) {
  Write-Host "Installing Visual Studio 2022 Build Tools C++ workload..." -ForegroundColor Cyan
  if ($winget) {
    winget install --id Microsoft.VisualStudio.2022.BuildTools --source winget --accept-package-agreements --accept-source-agreements --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  } else {
    choco install visualstudio2022buildtools visualstudio2022-workload-vctools -y --package-parameters "--includeRecommended"
  }
}

if ($InstallCMake) {
  Write-Host "Installing CMake..." -ForegroundColor Cyan
  if ($winget) {
    winget install --id Kitware.CMake --source winget --accept-package-agreements --accept-source-agreements
  } else {
    choco install cmake -y --installargs 'ADD_CMAKE_TO_PATH=System'
  }
}

Write-Host "C++ prerequisites requested. If this was the first install, open a new PowerShell window before building." -ForegroundColor Green
Write-Host "Gurobi is not installed by this script; install Gurobi separately and make sure GUROBI_HOME points to its win64 folder." -ForegroundColor Yellow
