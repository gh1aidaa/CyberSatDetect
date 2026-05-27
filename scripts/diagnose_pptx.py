import zipfile, re
z = zipfile.ZipFile(r"C:\Users\mohan\Desktop\CyberSatDetect_Poster_Academic.pptx")
xml = z.read("ppt/slides/slide1.xml").decode("utf-8")
sus = 0
for m in re.finditer(r'cx="(-?\d+)"\s+cy="(-?\d+)"', xml):
    cx, cy = int(m.group(1)), int(m.group(2))
    if cx <= 0 or cy <= 0 or cx > 100000000 or cy > 100000000:
        print("Suspicious dims:", cx, cy, "at offset", m.start())
        sus += 1
print("sp count:", xml.count("<p:sp>"))
print("pic count:", xml.count("<p:pic>"))
print("xml size:", len(xml))
print("sus dims:", sus)
# Also check for off positions
for m in re.finditer(r'<a:off\s+x="(-?\d+)"\s+y="(-?\d+)"', xml):
    x, y = int(m.group(1)), int(m.group(2))
    if x < -100000 or y < -100000 or x > 30270000 or y > 42813000 + 1000000:
        print("Off pos out of bounds:", x, y)
