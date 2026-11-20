param(
    [switch]$KeepOpen
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$labviewPath = 'F:\0_MyProject\LabView2018\LabVIEW 2021\LabVIEW.exe'
$viPath = Join-Path $scriptDir 'hello.vi'

if (-not (Test-Path $labviewPath)) {
    throw "LabVIEW 2021 not found: $labviewPath"
}

if (-not (Test-Path $viPath)) {
    throw "hello.vi not found: $viPath`nRun create_hello_vi.py first."
}

$arguments = @($viPath)
if ($KeepOpen) {
    $arguments += '-run'
} else {
    $arguments += '-run_then_exit'
}

Start-Process -FilePath $labviewPath -ArgumentList $arguments -Wait
Write-Host "Ran $viPath"
