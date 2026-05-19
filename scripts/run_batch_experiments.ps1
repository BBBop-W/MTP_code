param(
  [string]$RepoRoot = (Resolve-Path ".").Path,
  [string]$CondaEnv = "mtp",
  [string]$DateTag = (Get-Date -Format "yyyy-MM-dd"),
  [int]$ParallelJobs = 1,
  [int]$NumSplits = 3,
  [double]$TimeLimit = 3600.0,
  [string]$ProfileGeneratorMode = "gr",
  [switch]$SkipHeuristics,
  [switch]$NoWarmstart,
  [switch]$NoGenerate,
  [switch]$NoPipInstall
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
  Write-Host "conda not found. Install Miniconda/Anaconda first." -ForegroundColor Red
  exit 1
}

$exePath = Join-Path $RepoRoot "VNS_cpp\vns_solver.exe"
if (-not (Test-Path $exePath)) {
  Write-Host "VNS executable not found: $exePath" -ForegroundColor Red
  Write-Host "You said it has already been compiled. Please copy vns_solver.exe to VNS_cpp\vns_solver.exe." -ForegroundColor Yellow
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
  conda run -n $CondaEnv python "$RepoRoot/src/utility/batch_experiments.py" `
    --date $DateTag `
    --generate `
    --num-splits $NumSplits `
    --time-limit $TimeLimit `
    --profile-generator-mode $ProfileGeneratorMode
}

$extraArgs = @()
if ($SkipHeuristics) {
  $extraArgs += "--skip-heuristics"
}
if ($NoWarmstart) {
  $extraArgs += "--no-warmstart"
}

Write-Host "Starting $ParallelJobs experiment shard(s)..." -ForegroundColor Cyan

$jobs = @()

for ($sid = 0; $sid -lt $ParallelJobs; $sid++) {
  $jobName = "batch-$DateTag-shard-$sid"

  $jobs += Start-Job -Name $jobName -ScriptBlock {
    param(
      $RepoRoot,
      $CondaEnv,
      $DateTag,
      $ParallelJobs,
      $ShardId,
      $NumSplits,
      $TimeLimit,
      $ProfileGeneratorMode,
      $ExtraArgs
    )

    $ErrorActionPreference = "Stop"

    $argsList = @(
      "$RepoRoot/src/utility/batch_experiments.py",
      "--date", $DateTag,
      "--run",
      "--num-shards", "$ParallelJobs",
      "--shard-id", "$ShardId",
      "--result-suffix", "shard$ShardId",
      "--num-splits", "$NumSplits",
      "--time-limit", "$TimeLimit",
      "--profile-generator-mode", "$ProfileGeneratorMode"
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
    $ProfileGeneratorMode, `
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
$mergedCsv = "$resultDir/batch_results_$DateTag.csv"

$shardFiles = Get-ChildItem -Path $resultDir -Filter "batch_results_${DateTag}_shard*.csv" | Sort-Object Name

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

Write-Host "Merged results written to: $mergedCsv" -ForegroundColor Green
