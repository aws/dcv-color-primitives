# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

param([String]$filter="")

[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::TreatControlCAsInput = $true

$pwd = pwd
$exec = cargo bench --features std --no-run 2>&1 | select-string -Pattern 'benches\-[0-9a-f]+\.exe' | ForEach-Object { $_.Matches } | ForEach-Object { $_.Value }

$resultPath = 'target\criterion\*'
if (Test-Path -Path $resultPath) {
    Remove-Item $resultPath -Recurse -Force
}

$info = New-Object System.Diagnostics.ProcessStartInfo
$info.FileName = "$pwd\target\release\deps\$exec"
$info.RedirectStandardError = $false
$info.RedirectStandardOutput = $true
$info.UseShellExecute = $false
$info.Arguments = "--bench -n $filter"

$bench = New-Object System.Diagnostics.Process
$bench.StartInfo = $info

# Add event hander for stdout processing
$onOutputData = {
    if (($EventArgs.Data -ne $null) -and ($EventArgs.Data -match 'time:|thrpt:')) {
        Write-Host $EventArgs.Data
    }
}

$evt = Register-ObjectEvent -InputObject $bench `-Action $onOutputData -EventName 'OutputDataReceived'`

try {
    $bench.Start() | Out-Null

    # Begin benchmark output filtering
    $bench.BeginOutputReadLine()
    # Set process class to 'High'
    $bench.PriorityClass = 128
    # Set processor affinity to 'CPU 0'
    $bench.ProcessorAffinity = 1

    while (-not $bench.hasExited) {
        if ([Console]::KeyAvailable) {
            $readkey = [Console]::ReadKey($true)
            if ($readkey.Modifiers -eq 'Control' -and $readkey.Key -eq 'C') {
                Stop-Process -Force -InputObject $bench
                break
            }
        }

        Start-Sleep -Seconds 1
    }
} catch {
    Write-Host "An error occured while launching the executable"
} finally {
    # Unregister event handler for stdout processing
    Unregister-Event -SourceIdentifier $evt.Name
}
