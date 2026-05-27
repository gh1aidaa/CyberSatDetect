$ErrorActionPreference = 'Stop'

$pptxPath = 'C:\Users\mohan\Desktop\CyberSatDetect_Poster_A0_Final.pptx'
$pdfPath  = 'C:\Users\mohan\Desktop\CyberSatDetect_Poster_A0_Final.pdf'

$pp = New-Object -ComObject PowerPoint.Application
try { $pp.Visible = 1 } catch {}
try {
    $present = $pp.Presentations.Open($pptxPath, 0, 0, 0)
    try {
        # ppFixedFormatTypePDF = 2
        $present.SaveAs($pdfPath, 32)  # 32 = ppSaveAsPDF
        Write-Host "Saved PDF: $pdfPath"
    } finally {
        $present.Close()
    }
} finally {
    $pp.Quit()
}
