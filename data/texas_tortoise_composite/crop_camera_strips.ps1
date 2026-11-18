# Trim the Reconyx info strips (date/temperature bar on top; camera number and
# logo at the bottom) from the NACTI trail-camera frames, so that "has a black
# strip" cannot stand in for "no tortoise". Uses only Windows' built-in
# System.Drawing. Safe to re-run: only untouched 2048x1536 frames are cropped.
param([string]$Root = $PSScriptRoot)
Add-Type -AssemblyName System.Drawing
$top = 36; $bottom = 64
$codec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq 'image/jpeg' }
$params = New-Object System.Drawing.Imaging.EncoderParameters 1
$params.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter ([System.Drawing.Imaging.Encoder]::Quality, [long]95)
$done = 0; $skipped = 0
Get-ChildItem -Path (Join-Path $Root 'train'), (Join-Path $Root 'test') -Recurse -Filter 'nacti-*.jpg' | ForEach-Object {
    $src = [System.Drawing.Bitmap]::new($_.FullName)
    try {
        if ($src.Width -ne 2048 -or $src.Height -ne 1536) { $skipped++; return }
        $rect = [System.Drawing.Rectangle]::new(0, $top, $src.Width, $src.Height - $top - $bottom)
        $crop = $src.Clone($rect, $src.PixelFormat)
    } finally { $src.Dispose() }
    $tmp = $_.FullName + '.part'
    $crop.Save($tmp, $codec, $params); $crop.Dispose()
    Move-Item -LiteralPath $tmp -Destination $_.FullName -Force
    $done++
}
"cropped $done, already cropped or other size $skipped"
