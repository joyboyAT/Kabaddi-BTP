$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$annotationDir = Join-Path $projectRoot 'annotations'
$backupRoot = Join-Path $projectRoot ('annotations_backup_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
$reportPath = Join-Path $annotationDir 'class_id_report.csv'

if (-not (Test-Path $annotationDir)) {
    throw "Missing annotations folder: $annotationDir"
}

$classMap = @{
    # Fill in the mapping from existing YOLO class IDs to the new 0..5 class IDs.
    # Example: 0 = 0; 1 = 1; 3 = 2
    # Leave only the IDs that appear in your existing annotations.
}

$txtFiles = Get-ChildItem $annotationDir -Filter '*.txt' | Where-Object { $_.Name -ne 'classes.txt' }
$observedIds = New-Object System.Collections.Generic.HashSet[int]
$idCounts = @{}

foreach ($file in $txtFiles) {
    foreach ($line in Get-Content $file.FullName) {
        if ($line -match '^(\d+)\s+') {
            $id = [int]$matches[1]
            [void]$observedIds.Add($id)
            if (-not $idCounts.ContainsKey($id)) {
                $idCounts[$id] = 0
            }
            $idCounts[$id]++
        }
    }
}

$missing = @($observedIds | Where-Object { -not $classMap.ContainsKey($_) } | Sort-Object)
if ($missing.Count -gt 0) {
    $missingList = ($missing -join ', ')
    throw "Class map is incomplete. Add mappings for these existing IDs before running: $missingList"
}

New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null

Copy-Item (Join-Path $annotationDir '*.txt') $backupRoot -Force

foreach ($file in $txtFiles) {
    $newLines = foreach ($line in Get-Content $file.FullName) {
        if ($line -match '^(\d+)\s+(.*)$') {
            $oldId = [int]$matches[1]
            $rest = $matches[2]
            $newId = $classMap[$oldId]
            "$newId $rest"
        }
        else {
            $line
        }
    }

    Set-Content -Path $file.FullName -Value $newLines
}

$reportLines = @('old_id,count')
foreach ($id in ($idCounts.Keys | Sort-Object)) {
    $reportLines += "$id,$($idCounts[$id])"
}
Set-Content -Path $reportPath -Value $reportLines

Write-Host "Remap complete. Backup: $backupRoot"
Write-Host "Report written to: $reportPath"
