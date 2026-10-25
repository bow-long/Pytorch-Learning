$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Source = Join-Path $Root "programs\key_lock_launcher.py"
$Checkpoint = Join-Path $Root "tensor.pt"
$BuildRoot = Join-Path $Root "build"
$StageRoot = Join-Path $BuildRoot "portable-source"
$StageScript = Join-Path $StageRoot "key_lock_launcher.py"
$DistRoot = Join-Path $Root "dist"
$PortableRoot = Join-Path $DistRoot "portable"
$PortableApp = Join-Path $PortableRoot "KeyLockGame"
$Archive = Join-Path $DistRoot "KeyLockGame-portable.zip"
$GameFiles = @(
    "key_lock_game.py",
    "train_key_lock_game.py",
    "play_best_generation.py",
    "key_lock_launcher.py"
)

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Create .venv and install requirements-keyLockGame.txt first."
}
foreach ($GameFile in $GameFiles) {
    if (-not (Test-Path (Join-Path $Root "programs\$GameFile"))) {
        throw "Required game file was not found: programs\$GameFile"
    }
}
if (-not (Test-Path $Checkpoint)) {
    throw "Model checkpoint was not found: $Checkpoint"
}

if (Test-Path $PortableRoot) {
    Remove-Item -LiteralPath $PortableRoot -Recurse -Force
}
if (Test-Path $StageRoot) {
    Remove-Item -LiteralPath $StageRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $StageRoot, $PortableRoot | Out-Null
Copy-Item -LiteralPath $Source -Destination $StageScript -Force
foreach ($GameFile in $GameFiles) {
    Copy-Item -LiteralPath (Join-Path $Root "programs\$GameFile") -Destination $StageRoot -Force
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --windowed `
    --name KeyLockGame `
    --add-data "$Checkpoint;." `
    --distpath $PortableRoot `
    --workpath (Join-Path $BuildRoot "pyinstaller") `
    --specpath (Join-Path $BuildRoot "spec") `
    $StageScript
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}

if (-not (Test-Path (Join-Path $PortableApp "KeyLockGame.exe"))) {
    throw "PyInstaller did not create the expected executable."
}

foreach ($GameFile in $GameFiles) {
    Copy-Item -LiteralPath (Join-Path $Root "programs\$GameFile") -Destination $PortableApp -Force
}
Copy-Item -LiteralPath (Join-Path $Root "README.md") -Destination $PortableRoot -Force
@'
@echo off
"%~dp0KeyLockGame\KeyLockGame.exe" --mode train
if errorlevel 1 pause
'@ | Set-Content -LiteralPath (Join-Path $PortableRoot "Train-KeyLockGame.bat") -Encoding ASCII
@'
@echo off
"%~dp0KeyLockGame\KeyLockGame.exe" --mode play-best
if errorlevel 1 pause
'@ | Set-Content -LiteralPath (Join-Path $PortableRoot "Play-Best-Generation.bat") -Encoding ASCII
if (Test-Path $Archive) {
    Remove-Item -LiteralPath $Archive -Force
}
Compress-Archive -Path (Join-Path $PortableRoot "*") -DestinationPath $Archive -CompressionLevel Optimal

Write-Host "Portable folder: $PortableApp"
Write-Host "Training launcher: $(Join-Path $PortableRoot 'Train-KeyLockGame.bat')"
Write-Host "Best-generation launcher: $(Join-Path $PortableRoot 'Play-Best-Generation.bat')"
Write-Host "Portable archive: $Archive"
