"""Side-by-side sheets: real shorthand crop | font rendering of the label tokens | Roman word.
Used to test Le Jeune's Roman-spelling -> shorthand correspondence row by row."""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont
from chinukpipa.translit import tokens_to_unicode

# Path to a Duployan font with OpenType shaping (e.g. RawndMusmusDuployan-Regular.otf); set CHINUKPIPA_FONT.
FONT = os.environ.get('CHINUKPIPA_FONT', 'fonts/RawndMusmusDuployan-Regular.otf')

def render(tokens, h=56):
    f = ImageFont.truetype(FONT, 40, layout_engine=ImageFont.Layout.RAQM)
    s = tokens_to_unicode(tokens)
    im = Image.new('L', (600, 200), 255); d = ImageDraw.Draw(im)
    d.text((20, 60), s, font=f, fill=0)
    bb = im.point(lambda p: 255 - p).getbbox() or (0, 0, 10, 10)
    im = im.crop((bb[0] - 4, bb[1] - 4, bb[2] + 4, bb[3] + 4))
    sc = h / im.height
    return im.resize((max(1, int(im.width * sc)), h))

def sheets(jsonl, crops, outdir, field='tokens_rule', per=40, H=52):
    rows = [json.loads(l) for l in open(jsonl, encoding='utf-8')]
    rows = [r for r in rows if r.get(field)]
    os.makedirs(outdir, exist_ok=True)
    font = ImageFont.load_default(size=18); outs = []
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        sheet = Image.new('L', (1400, len(chunk) * (H + 8) + 10), 255); d = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            y = 5 + k * (H + 8)
            d.text((5, y + 15), str(s + k), fill=0, font=font)
            im = Image.open(os.path.join(crops, f"{r['id']}_shorthand.png")).convert('L')
            im = im.resize((max(1, min(300, int(im.width * H / im.height))), H)); sheet.paste(im, (60, y))
            ren = render(r[field], H); sheet.paste(ren.crop((0, 0, min(ren.width, 360), H)), (390, y))
            d.text((780, y + 15), f"{r['latin']}  [{r[field]}]", fill=0, font=font)
            d.line((0, y + H + 3, 1400, y + H + 3), fill=210)
        fn = os.path.join(outdir, f'compare_{s:03d}.png'); sheet.save(fn); outs.append(fn)
    return outs

if __name__ == '__main__':
    print('\n'.join(sheets(sys.argv[1], sys.argv[2], sys.argv[3])))
