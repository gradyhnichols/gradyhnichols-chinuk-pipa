"""Contact sheet of all pages in an IA *_orig_jp2.tar: python contact_sheet.py ITEM_DIR OUT.png [cols] [thumb]"""
import sys, tarfile, io, os
from PIL import Image, ImageDraw
item_dir, out = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 8
th = int(sys.argv[4]) if len(sys.argv) > 4 else 220
item = os.path.basename(os.path.normpath(item_dir))
tp = os.path.join(item_dir, f"{item}_orig_jp2.tar")
with tarfile.open(tp) as t:
    names = sorted(m.name for m in t.getmembers() if m.name.lower().endswith('.jp2'))
    thumbs = []
    for n in names:
        im = Image.open(io.BytesIO(t.extractfile(n).read())); im.draft('L', (th*2, th*2)); im = im.convert('L')
        im.thumbnail((th, th)); thumbs.append(im)
rows = (len(thumbs)+cols-1)//cols
sheet = Image.new('L', (cols*th, rows*(th+14)), 255); d = ImageDraw.Draw(sheet)
for i, im in enumerate(thumbs):
    x, y = (i % cols)*th, (i//cols)*(th+14)
    sheet.paste(im, (x, y+14)); d.text((x+2, y), str(i), fill=0)
sheet.save(out); print(len(names), sheet.size)
