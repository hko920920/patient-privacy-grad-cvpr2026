param(
    [string]$EnvironmentPath = ".venv-tfprivacy-baseline"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
& python (Join-Path $projectRoot "scripts\setup_tfprivacy_baseline.py") `
    --environment-path $EnvironmentPath
if ($LASTEXITCODE -ne 0) {
    throw "Cross-platform TensorFlow Privacy environment setup failed"
}
