$ErrorActionPreference = 'Stop'
$pp = New-Object -ComObject PowerPoint.Application
try { $pp.Visible = 1 } catch {}
try {
    $x = $pp.Presentations.Open('C:\Users\mohan\Desktop\test_blank.pptx', 0, 0, 0)
    Write-Host ('OPENED OK, slides: ' + $x.Slides.Count)
    $x.Close()
} catch {
    Write-Host ('OPEN FAILED: ' + $_)
} finally {
    $pp.Quit()
}
