[CmdletBinding()]
param(
    [string]$PythonExe = "python",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$projectFile = Join-Path $repoRoot "pyproject.toml"
if (-not (Test-Path -LiteralPath $projectFile -PathType Leaf)) {
    throw "Repository root is invalid: $repoRoot"
}

function Reset-RepoDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $repoPrefix = $repoRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) +
        [System.IO.Path]::DirectorySeparatorChar
    if (
        -not $fullPath.StartsWith(
            $repoPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )
    ) {
        throw "Refusing to clear a path outside the repository: $fullPath"
    }
    if (Test-Path -LiteralPath $fullPath) {
        Remove-Item -LiteralPath $fullPath -Recurse -Force
    }
    New-Item -ItemType Directory -Path $fullPath | Out-Null
}

$projectText = Get-Content -LiteralPath $projectFile -Raw
$versionMatch = [regex]::Match(
    $projectText,
    '(?m)^version\s*=\s*"([^"]+)"\s*$'
)
if (-not $versionMatch.Success) {
    throw "Cannot read project version from $projectFile"
}
$version = $versionMatch.Groups[1].Value

$buildRoot = Join-Path $repoRoot "build\pyinstaller"
$releaseRoot = Join-Path $repoRoot "release"
$bundleName = "vllm-ascend-hbm-windows-x64-v$version"
$bundleRoot = Join-Path $releaseRoot $bundleName
$archivePath = Join-Path $releaseRoot "$bundleName.zip"
$specFile = Join-Path $repoRoot "packaging\pyinstaller\vllm_ascend_hbm.spec"

Reset-RepoDirectory -Path $buildRoot
Reset-RepoDirectory -Path $releaseRoot
New-Item -ItemType Directory -Path $bundleRoot | Out-Null

Push-Location $repoRoot
try {
    if (-not $SkipTests) {
        $previousPythonPath = $env:PYTHONPATH
        try {
            $env:PYTHONPATH = Join-Path $repoRoot "src"
            & $PythonExe -m unittest discover -s tests -p "test_*.py"
            if ($LASTEXITCODE -ne 0) {
                throw "Unit tests failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            $env:PYTHONPATH = $previousPythonPath
        }
    }

    & $PythonExe -m PyInstaller `
        --noconfirm `
        --clean `
        --workpath $buildRoot `
        --distpath $bundleRoot `
        $specFile
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE"
    }

    Copy-Item `
        -LiteralPath (Join-Path $repoRoot "configs") `
        -Destination $bundleRoot `
        -Recurse
    Copy-Item `
        -LiteralPath (Join-Path $repoRoot "docs\EXE_USAGE.md") `
        -Destination $bundleRoot
    Copy-Item `
        -LiteralPath (Join-Path $repoRoot "README.md") `
        -Destination $bundleRoot

    $exePath = Join-Path $bundleRoot "vllm-ascend-hbm.exe"
    if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
        throw "Expected executable was not generated: $exePath"
    }

    & $exePath --help | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "EXE --help smoke test failed"
    }
    & $exePath --list-models | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "EXE --list-models smoke test failed"
    }

    $hardwareConfig = Join-Path $bundleRoot "configs\hardware_910c_1node.json"
    $inferenceConfig = Join-Path $bundleRoot "configs\deepseek_v4_flash_910c_inference.json"
    $commonArguments = @(
        "--hardware-config", $hardwareConfig,
        "--config", $inferenceConfig,
        "--operation", "estimate",
        "--format", "json"
    )
    $exeJson = (& $exePath @commonArguments) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0) {
        throw "EXE dual-JSON estimate smoke test failed"
    }
    $sourceJson = (
        & $PythonExe `
            (Join-Path $repoRoot "vllm_ascend_hbm_calculator.py") `
            @commonArguments
    ) -join [Environment]::NewLine
    if ($LASTEXITCODE -ne 0) {
        throw "Source CLI comparison run failed"
    }
    if ($exeJson -ne $sourceJson) {
        throw "EXE output differs from the Python CLI output"
    }
    $parsed = $exeJson | ConvertFrom-Json
    if (
        $parsed.config.platform.logical_device_count -ne 16 -or
        $parsed.config.platform.visible_hbm_gib_per_die -ne 61.27
    ) {
        throw "EXE did not apply the separate hardware configuration"
    }

    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $exePath
    "$($hash.Hash.ToLowerInvariant())  vllm-ascend-hbm.exe" |
        Set-Content `
            -LiteralPath (Join-Path $bundleRoot "SHA256SUMS.txt") `
            -Encoding ascii

    Compress-Archive `
        -Path (Join-Path $bundleRoot "*") `
        -DestinationPath $archivePath `
        -CompressionLevel Optimal

    Write-Output "Executable: $exePath"
    Write-Output "Archive:    $archivePath"
    Write-Output "SHA256:     $($hash.Hash.ToLowerInvariant())"
}
finally {
    Pop-Location
}
