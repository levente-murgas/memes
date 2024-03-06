$scriptName = "cleanimages.py" # replace with your script name
$pythonProcesses = Get-Process python

$flag = 0
foreach ($process in $pythonProcesses) {
    $commandLine = (Get-WmiObject Win32_Process -Filter "Handle = $($process.Id)").CommandLine
    if ($commandLine -like "*$scriptName*") {
        Write-Output "The script '$scriptName' is running with PID $($process.Id)."
        $flag = 1
    }
}
Write-Output "Flag: $flag"
