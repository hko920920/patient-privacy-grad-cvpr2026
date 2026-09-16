$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    python -m json.tool docs\privacy_report_schema_v2_0.json | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "privacy_report_schema_v2_0.json is not valid JSON."
    }

    python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "v2.0 regression suite failed."
    }

    Write-Host "v2.0 schema and regression suite passed."
}
finally {
    Pop-Location
}
