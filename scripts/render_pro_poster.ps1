$ErrorActionPreference = 'Stop'

$pptxPath = 'C:\Users\mohan\Desktop\CyberSatDetect_Poster_EN_Pro.pptx'
$outDir   = 'C:\Users\mohan\Desktop\gh\CyberSatDetectprojct2\CyberSatDetectprojct\scripts\poster_preview'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$pp = New-Object -ComObject PowerPoint.Application
try { $pp.Visible = 1 } catch {}
try {
    $present = $pp.Presentations.Open($pptxPath, 0, 0, 0)
    try {
        $widthPx  = 1700
        $heightPx = [int]($widthPx * (3371 / 2384))

        $outPng = Join-Path $outDir 'pro_poster_page1.png'
        $present.Slides.Item(1).Export($outPng, 'png', $widthPx, $heightPx)
        Write-Host "Saved: $outPng  ($widthPx x $heightPx)"

        $pdfPath = 'C:\Users\mohan\Desktop\CyberSatDetect_Poster_EN_Pro.pdf'
        $present.SaveAs($pdfPath, 32)
        Write-Host "Saved PDF: $pdfPath"
    } finally {
        $present.Close()
    }
} finally {
    $pp.Quit()
}
