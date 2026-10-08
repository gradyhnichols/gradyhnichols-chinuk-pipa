"""Extract page images from an Internet Archive *_orig_jp2.tar (cached locally).

usage: python ia_page.py ITEM_DIR PAGE[,PAGE...] OUTDIR [--max 1600]
Pages are 0-based indexes into the sorted list of JP2 members.
"""
import sys, tarfile, io, os
from PIL import Image

def members(tar_path):
    with tarfile.open(tar_path) as t:
        return sorted(m.name for m in t.getmembers() if m.name.lower().endswith('.jp2'))

def extract(item_dir, pages, outdir, maxside=None):
    item = os.path.basename(os.path.normpath(item_dir))
    tp = os.path.join(item_dir, f"{item}_orig_jp2.tar")
    names = members(tp)
    os.makedirs(outdir, exist_ok=True)
    out = []
    with tarfile.open(tp) as t:
        for p in pages:
            im = Image.open(io.BytesIO(t.extractfile(names[p]).read()))
            im.load()
            if maxside:
                im.thumbnail((maxside, maxside))
            fn = os.path.join(outdir, f"{item}_p{p:03d}.png")
            im.convert('L').save(fn)
            out.append((fn, im.size))
    return names, out

if __name__ == '__main__':
    item_dir, pages, outdir = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
    mx = int(sys.argv[sys.argv.index('--max')+1]) if '--max' in sys.argv else None
    names, out = extract(item_dir, pages, outdir, mx)
    print(len(names), 'pages in item')
    for o in out: print(*o)
