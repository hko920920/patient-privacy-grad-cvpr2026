param(
    [double]$MinimumFreeGiB = 50.0,
    [double]$MaximumWorkingGiB = 15.0
)

$ErrorActionPreference = 'Stop'

$WorkingRoot = Split-Path -Parent $PSScriptRoot
$OriginalsRoot = Join-Path (Split-Path -Parent $WorkingRoot) 'code_originals'
$DriveName = ([System.IO.Path]::GetPathRoot($WorkingRoot)).TrimEnd([char]92).TrimEnd(':')
$Drive = Get-PSDrive -Name $DriveName

function Get-TreeBytes {
    param([string]$LiteralRoot)

    if (-not (Test-Path -LiteralPath $LiteralRoot)) {
        return 0
    }

    $Files = Get-ChildItem -LiteralPath $LiteralRoot -Recurse -File -Force -ErrorAction SilentlyContinue
    $Measured = $Files | Measure-Object -Property Length -Sum
    if ($null -eq $Measured.Sum) {
        return 0
    }
    return [int64]$Measured.Sum
}

$WorkingBytes = Get-TreeBytes -LiteralRoot $WorkingRoot
$OriginalsBytes = Get-TreeBytes -LiteralRoot $OriginalsRoot
$FreeGiB = $Drive.Free / 1GB
$WorkingGiB = $WorkingBytes / 1GB
$ReviewRequired = ($FreeGiB -lt $MinimumFreeGiB) -or ($WorkingGiB -gt $MaximumWorkingGiB)

$Result = [ordered]@{
    status = if ($ReviewRequired) { 'REVIEW_REQUIRED' } else { 'OK' }
    checked_at = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
    drive = $DriveName
    free_gib = [math]::Round($FreeGiB, 2)
    minimum_free_gib = $MinimumFreeGiB
    code_originals_gib = [math]::Round($OriginalsBytes / 1GB, 3)
    code_working_gib = [math]::Round($WorkingGiB, 3)
    maximum_working_gib = $MaximumWorkingGiB
}

$Result | ConvertTo-Json
