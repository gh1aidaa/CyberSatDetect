$ErrorActionPreference = 'Stop'
$pp = New-Object -ComObject PowerPoint.Application
try { $pp.Visible = 1 } catch {}
try {
    $x = $pp.Presentations.Open('C:\Users\mohan\Desktop\test_copy.pptx', 0, 0, 0)
    Write-Host ('OK shapes: ' + $x.Slides.Item(1).Shapes.Count)
    $x.Close()
} catch {
    Write-Host ('FAIL: ' + $_)
} finally {
    $pp.Quit()
}
