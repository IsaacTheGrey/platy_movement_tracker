<#
.SYNOPSIS
    Stitch all “N.mp4” files (sorted numerically) into one “merged_all.mp4,”
    then re‐encode at a custom frame rate (e.g., 1 frame per 60 seconds), dropping audio.
    If OriginalFps is provided, adjusts playback speed to match real recording time.

.PARAMETER TargetFps
    Frames per second for the output. Default is 1/60 ≈ 0.016667.

.PARAMETER OriginalFps
    Actual frames per second of the input video, used for timestamp correction.

.EXAMPLE
    PS> .\StitchAndDownsample.ps1 -TargetFps 0.0166667 -OriginalFps 5
#>

param (
    [double] $TargetFps    = 0.0166667,
    [double] $OriginalFps
)

# Ensure FFmpeg is available
if (-not (Get-Command ffmpeg.exe -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg.exe not found in PATH. Install FFmpeg and add it to PATH."
    exit 1
}

# Step 1: Build list of MP4s in natural numeric order
$listFile = "ffmpeg_concat_list.txt"
if (Test-Path $listFile) { Remove-Item $listFile }

$mp4Files = Get-ChildItem -Filter '*.mp4' |
    Sort-Object {
        $base = [IO.Path]::GetFileNameWithoutExtension($_.Name)
        if ($base -as [int] -ne $null) { [int]$base } else { 0 }
    }

if ($mp4Files.Count -eq 0) {
    Write-Error "No .mp4 files found in $(Get-Location)."
    exit 1
}

foreach ($file in $mp4Files) {
    # FFmpeg on Windows accepts forward‐slashes in the list file
    $pathUnix = $file.FullName.Replace('\','/')
    Add-Content -Path $listFile -Value "file '$pathUnix'"
}

# Step 2: Concatenate without re-encoding, move moov atom to front
$merged = "merged_all.mp4"
Write-Host ">>> Concatenating in numeric order → $merged"
ffmpeg -f concat -safe 0 -i $listFile -c copy -movflags +faststart $merged -y

# Step 3: Re-encode at target FPS, drop audio, with optional PTS correction
$output = "downsampled.mp4"
Write-Host ">>> Re-encoding $merged at $TargetFps fps → $output"

# Build the video filter chain
$filter = "fps=$TargetFps"
if ($PSBoundParameters.ContainsKey("OriginalFps")) {
    $multiplier = [math]::Round($OriginalFps / $TargetFps, 6)
    $filter += ",setpts=${multiplier}*PTS"
    Write-Host ">>> Applying timestamp correction with multiplier: $multiplier"
}

# Try direct MP4 muxing with regenerated pts
ffmpeg -fflags +genpts -i $merged `
       -vf $filter `
       -vsync vfr `
       -c:v libx264 -preset medium -crf 23 `
       -an `
       $output -y

if ($LASTEXITCODE -ne 0) {
    Write-Warning "MP4 muxer failed—falling back to MKV"
    $mkv = "downsampled.mkv"
    ffmpeg -fflags +genpts -i $merged `
           -vf $filter -vsync vfr `
           -c:v libx264 -preset medium -crf 23 -an `
           $mkv -y

    Write-Host ">>> Remuxing MKV → MP4"
    ffmpeg -i $mkv -c copy -movflags +faststart $output -y
    Remove-Item $mkv
}

Write-Host "✅ Done. Final video → $output"
Write-Host "INFO: You can delete '$merged' if no longer needed."
