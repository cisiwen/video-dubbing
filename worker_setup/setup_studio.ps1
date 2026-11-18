# One-time setup for the studio on the GPU box (run by deploy_studio.sh --setup): its own small
# Python environment (the worker's torch environment is left alone), a portable ffmpeg, and
# scheduled tasks that start the worker and the studio at boot.
$ErrorActionPreference = "Stop"
$dir = "D:\dubbing-studio"
Set-Location $dir

if (-not (Test-Path .venv)) { uv venv --python 3.11 .venv }
uv pip install --python .venv\Scripts\python.exe -r requirements-studio.txt

if (-not (Test-Path "$dir\ffmpeg\bin\ffmpeg.exe")) {
    $zip = "$dir\ffmpeg.zip"
    curl.exe -sSL --retry 5 -o $zip "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    Expand-Archive -Force $zip "$dir\ffmpeg-tmp"
    $bin = Get-ChildItem "$dir\ffmpeg-tmp" -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    New-Item -ItemType Directory -Force "$dir\ffmpeg" | Out-Null
    Move-Item -Force $bin.Directory.FullName "$dir\ffmpeg\bin"
    Remove-Item -Recurse -Force "$dir\ffmpeg-tmp", $zip
}

# Both at boot, whether or not anyone is logged in (S4U: no stored password), restarted by
# their own loops. Not $env:USERDOMAIN\$env:USERNAME: over SSH that's "WORKGROUP".
$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
foreach ($t in @(@('DubbingWorker', 'D:\dubbing-worker\start_worker.ps1', 'D:\dubbing-worker'),
                 @('DubbingStudio', "$dir\start_studio.ps1", $dir))) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File $($t[1])" -WorkingDirectory $t[2]
    Register-ScheduledTask -TaskName $t[0] -Action $action -Trigger (New-ScheduledTaskTrigger -AtStartup) -Principal $principal -Settings $settings -Force | Out-Null
}
"studio set up in $dir"
