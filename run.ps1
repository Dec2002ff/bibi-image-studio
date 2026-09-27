<#
.SYNOPSIS
    Starts the studio in the project environment.

.DESCRIPTION
    The application runs only in its own .venv: the system Python has neither
    diffusers from git nor torch with CUDA. The script checks that the
    environment works, not merely that it exists: it may be left over from an
    interrupted installation.

    All arguments are passed to the application as is.

    The browser opens by itself once the server is ready to serve the page.
    The application opens it, not this script: the script does not know when
    the server is ready, and Python with torch and diffusers takes seconds to
    start, so a window opened right away would hit "can't connect". Disable
    it with --no-open-browser.

.EXAMPLE
    .\run.ps1

.EXAMPLE
    .\run.ps1 --lang ru --port 7870
#>

[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Arguments)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    Write-Host 'The .venv environment was not found.' -ForegroundColor Yellow
    Write-Host 'Create it once:' -ForegroundColor Yellow
    Write-Host '    .\install.ps1' -ForegroundColor Cyan
    exit 1
}

$check = & $python -c "import torch, sys; sys.stdout.write('cuda' if torch.cuda.is_available() else 'cpu')" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'The .venv environment is broken: torch cannot be imported.' -ForegroundColor Red
    Write-Host '    Remove-Item -Recurse -Force .venv; .\install.ps1' -ForegroundColor Cyan
    exit 1
}
if ($check -ne 'cuda') {
    Write-Host 'CUDA is not available: generation would run on the CPU and take hours.' -ForegroundColor Yellow
    Write-Host 'Reinstall torch:' -ForegroundColor Yellow
    Write-Host '    .venv\Scripts\python -m pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128' -ForegroundColor Cyan
}

# The flag goes first so that --no-open-browser from the user's arguments
# comes after it and wins: argparse takes the last value.
$launchArguments = @('--open-browser') + $Arguments

Push-Location $root
try {
    & $python -m fooocus_qwen @launchArguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
