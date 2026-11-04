
import docx, zipfile, os
p = r"C:\dev\图像处理大作业\lowlight-retinex\report\低光照图像增强大作业报告.docx"
d = docx.Document(p)
print("paragraphs", len(d.paragraphs), "tables", len(d.tables), "sections", len(d.sections))
print("inline shapes", len(d.inline_shapes))
text = "\n".join(x.text for x in d.paragraphs)
print("has nan:", "nan" in text.lower(), "has inf:", "inf" in text.lower())
print("chars", len(text))
z = zipfile.ZipFile(p)
imgs = [n for n in z.namelist() if n.startswith("word/media/")]
print("media", len(imgs), [ (n, z.getinfo(n).file_size) for n in imgs])
for i,t in enumerate(d.tables):
    print("table", i, len(t.rows), "x", len(t.columns), "first:", " | ".join(c.text for c in t.rows[0].cells))
