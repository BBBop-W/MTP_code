param(
  [string]$RepoRoot = (Resolve-Path ".").Path,
  [string]$CondaEnv = "mtp",
  [string]$DateTag = (Get-Date -Format "yyyy-MM-dd"),
  [int]$ParallelJobs = 1,
  [int]$NumSplits = 3,
  [double]$TimeLimit = 3600.0,
  [double]$MipGap = 0.0001,
  [string]$ProfileGeneratorMode = "hyb",
  [string]$ResidualProfileMode = "full",
  [int]$MaxNodes = 5000,
  [int]$MaxCgIters = 3000,
  [int]$LabelColumns = 20,
  [int]$SolverColumns = 1,
  [string[]]$Methods = @(),
  [string]$BpcExe = "",
  [string]$VnsExe = "",
  [string]$GurobiHome = "",
  [ValidateSet("x64", "Win32", "ARM64", "ARM64EC")]
  [string]$Architecture = "x64",
  [switch]$BuildCpp,
  [switch]$CleanCppBuild,
  [switch]$InstallCppPrereqs,
  [switch]$RefreshWarmstart,
  [switch]$NoGenerate,
  [switch]$NoPipInstall
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  Write-Host "conda not found. Install Miniconda/Anaconda first." -ForegroundColor Red
  exit 1
}

if ([string]::IsNullOrWhiteSpace($BpcExe)) {
  $BpcExe = Join-Path $RepoRoot "BPC_label_cpp\bpc_label_solver.exe"
}
if ([string]::IsNullOrWhiteSpace($VnsExe)) {
  $VnsExe = Join-Path $RepoRoot "VNS_cpp\vns_solver.exe"
}

if ($InstallCppPrereqs) {
  & "$RepoRoot/scripts/setup_cpp_windows.ps1" -InstallAll
  if (-not (Get-Command cmake -ErrorAction SilentlyContinue) -and (-not (Test-Path $BpcExe) -or -not (Test-Path $VnsExe))) {
    Write-Host "CMake is not visible in this PowerShell yet. Open a new PowerShell window and rerun this script." -ForegroundColor Yellow
    exit 0
  }
}

if ($BuildCpp -or -not (Test-Path $BpcExe) -or -not (Test-Path $VnsExe)) {
  Write-Host "Building required C++ executables..." -ForegroundColor Cyan
  $buildArgs = @(
    "-RepoRoot", $RepoRoot,
    "-Architecture", $Architecture
  )
  if (-not [string]::IsNullOrWhiteSpace($GurobiHome)) {
    $buildArgs += @("-GurobiHome", $GurobiHome)
  }
  if ($CleanCppBuild) {
    $buildArgs += "-Clean"
  }
  & "$RepoRoot/scripts/build_cpp_tools.ps1" @buildArgs
}

if (-not (Test-Path $BpcExe) -or -not (Test-Path $VnsExe)) {
  Write-Host "Required C++ executables are still missing after build attempt." -ForegroundColor Red
  Write-Host "BPC: $BpcExe" -ForegroundColor Yellow
  Write-Host "VNS: $VnsExe" -ForegroundColor Yellow
  Write-Host "If this is a fresh Windows server, run with -InstallCppPrereqs once, open a new PowerShell, then rerun." -ForegroundColor Yellow
  exit 1
}

$envList = conda env list | Out-String
if ($envList -notmatch "^\s*$CondaEnv\s") {
  conda create -y -n $CondaEnv python=3.10
}

if (-not $NoPipInstall) {
  conda run -n $CondaEnv python -m pip install --upgrade pip
  conda run -n $CondaEnv python -m pip install pandas numpy openpyxl gurobipy
}

if (-not $NoGenerate) {
  conda run -n $CondaEnv python "$RepoRoot/src/utility/cpp_core_batch_experiments.py" `
    --date $DateTag `
    --generate `
    --num-splits $NumSplits `
    --time-limit $TimeLimit `
    --mip-gap $MipGap `
    --profile-generator-mode $ProfileGeneratorMode `
    --residual-profile-mode $ResidualProfileMode
}

$extraArgs = @()
if ($RefreshWarmstart) {
  $extraArgs += "--refresh-warmstart"
}
if ($Methods.Count -gt 0) {
  $extraArgs += "--methods"
  foreach ($method in $Methods) {
    $extraArgs += $method
  }
}

Write-Host "Starting $ParallelJobs C++ core experiment shard(s)..." -ForegroundColor Cyan

$jobs = @()

for ($sid = 0; $sid -lt $ParallelJobs; $sid++) {
  $jobName = "cpp-core-$DateTag-shard-$sid"

  $jobs += Start-Job -Name $jobName -ScriptBlock {
    param(
      $RepoRoot,
      $CondaEnv,
      $DateTag,
      $ParallelJobs,
      $ShardId,
      $NumSplits,
      $TimeLimit,
      $MipGap,
      $ProfileGeneratorMode,
      $ResidualProfileMode,
      $MaxNodes,
      $MaxCgIters,
      $LabelColumns,
      $SolverColumns,
      $BpcExe,
      $VnsExe,
      $ExtraArgs
    )

    $ErrorActionPreference = "Stop"

    $argsList = @(
      "$RepoRoot/src/utility/cpp_core_batch_experiments.py",
      "--date", $DateTag,
      "--run",
      "--num-shards", "$ParallelJobs",
      "--shard-id", "$ShardId",
      "--result-suffix", "shard$ShardId",
      "--num-splits", "$NumSplits",
      "--time-limit", "$TimeLimit",
      "--mip-gap", "$MipGap",
      "--profile-generator-mode", "$ProfileGeneratorMode",
      "--residual-profile-mode", "$ResidualProfileMode",
      "--max-nodes", "$MaxNodes",
      "--max-cg-iters", "$MaxCgIters",
      "--label-columns", "$LabelColumns",
      "--solver-columns", "$SolverColumns",
      "--bpc-exe", "$BpcExe",
      "--vns-exe", "$VnsExe"
    )

    foreach ($arg in $ExtraArgs) {
      $argsList += $arg
    }

    conda run -n $CondaEnv python @argsList

  } -ArgumentList `
    $RepoRoot, `
    $CondaEnv, `
    $DateTag, `
    $ParallelJobs, `
    $sid, `
    $NumSplits, `
    $TimeLimit, `
    $MipGap, `
    $ProfileGeneratorMode, `
    $ResidualProfileMode, `
    $MaxNodes, `
    $MaxCgIters, `
    $LabelColumns, `
    $SolverColumns, `
    $BpcExe, `
    $VnsExe, `
    $extraArgs
}

Wait-Job $jobs

$failed = $false

foreach ($job in $jobs) {
  Write-Host "===== Output from $($job.Name) =====" -ForegroundColor Yellow
  Receive-Job $job

  if ($job.State -ne "Completed") {
    Write-Host "Job $($job.Name) failed with state $($job.State)" -ForegroundColor Red
    $failed = $true
  }
}

Remove-Job $jobs

if ($failed) {
  Write-Host "At least one shard failed. Please check the output above." -ForegroundColor Red
  exit 1
}

$resultDir = "$RepoRoot/result/batch_$DateTag"
$mergedCsv = "$resultDir/cpp_core_results_$DateTag.csv"

$shardFiles = Get-ChildItem -Path $resultDir -Filter "cpp_core_results_${DateTag}_shard*.csv" | Sort-Object Name

if ($shardFiles.Count -eq 0) {
  Write-Host "No shard result files found in $resultDir" -ForegroundColor Red
  exit 1
}

$allRows = @()
foreach ($file in $shardFiles) {
  Write-Host "Merging $($file.Name)" -ForegroundColor Cyan
  $allRows += Import-Csv $file.FullName
}

if (Test-Path $mergedCsv) {
  Remove-Item $mergedCsv -Force
}

$allRows |
  Sort-Object instance_id, method |
  Export-Csv $mergedCsv -NoTypeInformation -Encoding UTF8

Write-Host "Merged C++ core results written to: $mergedCsv" -ForegroundColor Green
