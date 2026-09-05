param(
    [Parameter(Mandatory=$true)][string]$Inputs,
    [string]$Python = "python",
    [string]$Output = "dist/windows",
    [string]$ISCC = "",
    [string]$Demo = "",
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepositoryRoot
try {
    $BuildArguments = @("-m", "tools.build_windows", "--inputs", $Inputs, "--output", $Output)
    if ($ValidateOnly) { $BuildArguments += "--validate-only" }
    if ($ISCC) { $BuildArguments += @("--iscc", $ISCC) }
    if ($Demo) { $BuildArguments += @("--demo", $Demo) }
    & $Python @BuildArguments
    if ($LASTEXITCODE -ne 0) { throw "Windows build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
