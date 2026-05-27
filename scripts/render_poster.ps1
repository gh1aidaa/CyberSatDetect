$ErrorActionPreference = 'Stop'

$pptxPath = 'C:\Users\mohan\Desktop\CyberSatDetect_Poster_A0_Final.pptx'
$outDir   = 'C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\scripts\poster_preview'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$pp = New-Object -ComObject PowerPoint.Application
# msoTrue = -1, msoFalse = 0
try { $pp.Visible = 1 } catch {}
try {
    $present = $pp.Presentations.Open($pptxPath, 0, 0, 0)
    try {
        # Render the slide to PNG at ~150 DPI for a 84.1 x 118.9 cm A0 portrait poster.
        # Slide size: 8401.7 x 11890 100ths of a mm? PowerPoint width/height are in points.
        # 1 cm = 28.3464567 pt. For A0 portrait: width  = 84.1 * 28.346 = 2384 pt; height = 118.9 * 28.346 = 3371 pt.
        # At 96 dpi the slide is 2384/72 * 96 = 3179 px wide; 3371/72*96 = 4495 px tall.
        # We'll request 2500 px wide which keeps file manageable.
        $widthPx  = 1700
        $heightPx = [int]($widthPx * (3371 / 2384))

        $outPng = Join-Path $outDir 'poster_page1.png'
        $present.Slides.Item(1).Export($outPng, 'png', $widthPx, $heightPx)
        Write-Host "Saved: $outPng  ($widthPx x $heightPx)"
    } finally {
        $present.Close()
    }
} finally {
    $pp.Quit()
}
