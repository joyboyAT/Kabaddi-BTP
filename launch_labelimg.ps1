$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$annotationDir = Join-Path $projectRoot 'annotations'
$annotationClasses = Join-Path $annotationDir 'classes.txt'
$masterClasses = Join-Path $projectRoot 'classes_master.txt'

if (-not (Test-Path $annotationDir)) {
    throw "Missing annotations folder: $annotationDir"
}

if (-not (Test-Path $masterClasses)) {
    if (Test-Path $annotationClasses) {
        Copy-Item $annotationClasses $masterClasses -Force
    }
    else {
        throw "No class list found. Create either $masterClasses or $annotationClasses first."
    }
}

if (-not (Test-Path $annotationClasses)) {
    Copy-Item $masterClasses $annotationClasses -Force
}
else {
    Copy-Item $masterClasses $annotationClasses -Force
}

if (-not (Test-Path $masterClasses)) {
    throw "No class list found. Create either $masterClasses or $annotationClasses first."
}

$labelImgExeCandidates = @(
    (Join-Path $env:APPDATA 'Python\Python313\Scripts\labelImg.exe'),
    (Join-Path $projectRoot '.venv\Scripts\labelImg.exe'),
    (Join-Path $projectRoot '.venv\Scripts\python.exe')
)

$labelImgExe = $null
foreach ($candidate in $labelImgExeCandidates) {
    if (Test-Path $candidate) {
        $labelImgExe = $candidate
        break
    }
}

if (-not $labelImgExe) {
    throw 'Could not find a LabelImg launcher.'
}

$imageDir = Join-Path $projectRoot 'frames'

Push-Location $projectRoot
try {
    if ($labelImgExe.EndsWith('python.exe')) {
        & $labelImgExe -m labelImg $imageDir $masterClasses $annotationDir
    }
    else {
        & $labelImgExe $imageDir $masterClasses $annotationDir
    }
}
finally {
    Pop-Location
}

Copy-Item $masterClasses $annotationClasses -Force