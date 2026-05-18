param(
  [string]$RepoRoot = (Resolve-Path ".").Path,
  [string]$CondaEnv = "mtp",
  [string]$DateTag = (Get-Date -Format "yyyy-MM-dd")
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  Write-Host "conda not found. Install Miniconda/Anaconda first." -ForegroundColor Red
  exit 1
}

conda create -y -n $CondaEnv python=3.10
conda activate $CondaEnv

python -m pip install --upgrade pip
python -m pip install pandas numpy openpyxl gurobipy

# Ensure Gurobi is installed and licensed on this machine.
# Set GUROBI_HOME and license if needed before running.

cmake -S "$RepoRoot/VNS_cpp" -B "$RepoRoot/VNS_cpp/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$RepoRoot/VNS_cpp/build" --config Release

$exeCandidates = @(
  "$RepoRoot/VNS_cpp/build/Release/vns_solver.exe",
  "$RepoRoot/VNS_cpp/build/vns_solver.exe"
)
foreach ($exe in $exeCandidates) {
  if (Test-Path $exe) {
    Copy-Item $exe "$RepoRoot/VNS_cpp/vns_solver.exe" -Force
    break
  }
}

python "$RepoRoot/src/utility/batch_experiments.py" --date $DateTag --generate --run
