# Define the path to your Python script
$preprocess_script = "cleanimages.py"
$check_script = "./check_progress.py"

while ($true) {
    # Try to get a process which is your Python script
    $pythonProcesses = Get-Process python

    $flag = 0
    foreach ($process in $pythonProcesses) {
        $commandLine = (Get-WmiObject Win32_Process -Filter "Handle = $($process.Id)").CommandLine
        if ($commandLine -like "*$preprocess_script*") {
            Write-Output "The script '$preprocess_script' is running with PID $($process.Id)."
            $flag = 1
        }
    }

    if ($flag -eq 0) {
        Write-Output "Python script not running. Checking progress..."
        Start-Process "python" -ArgumentList $check_script
        $progress = Get-Content progress.txt -TotalCount 1
        Write-Output "Progress: $progress"
        if ($progress -ne 410660) {
            Write-Output "Restarting Python script..."
            Start-Process "python" -ArgumentList $preprocess_script
            Write-Output "Python script started."
        }
        else {
            Write-Output "All images have been processed."
            break
        }
    }
    
    Start-Sleep -Seconds 5
}
