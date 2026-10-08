"""Numbered review sheets: [#] [Roman crop] [shorthand crop] [gloss crop]  text -> for checking readings by eye.

python -m chinukpipa.gt.review_sheet ROWS.jsonl CROPS_DIR OUT_DIR
ROWS.jsonl rows need `id` and either `latin_ocr`/`gloss_ocr` (harvest proposals) or `latin`/`gloss_en`
(annotations); crops are `<id>_{latin,shorthand,gloss}.png` as written by harvest_vocab or make_crops.
"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

def sheets(jsonl, crops, outdir, per=30, H=56):
    rows = [json.loads(l) for l in open(jsonl, encoding='utf-8')]
    os.makedirs(outdir, exist_ok=True)
    font = ImageFont.load_default(size=20)
    out = []
    for s in range(0, len(rows), per):
        chunk = rows[s:s + per]
        sheet = Image.new('L', (1500, per * (H + 8) + 10), 255)
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(chunk):
            y = 5 + k * (H + 8)
            d.text((5, y + 15), str(s + k), fill=0, font=font)
            x = 60
            for part, maxw in (('latin', 330), ('shorthand', 260), ('gloss', 330)):
                im = Image.open(os.path.join(crops, f"{r['id']}_{part}.png")).convert('L')
                sc = H / im.height
                im = im.resize((max(1, min(maxw, int(im.width * sc))), H))
                sheet.paste(im, (x, y)); x += maxw + 15
            lat = r.get('latin_ocr', r.get('latin', '')); glo = r.get('gloss_ocr', r.get('gloss_en', ''))
            d.text((x, y + 15), f"{lat} | {glo}", fill=0, font=font)
            d.line((0, y + H + 3, 1500, y + H + 3), fill=200)
        fn = os.path.join(outdir, f'review_{s:03d}.png'); sheet.save(fn); out.append(fn)
    return out

if __name__ == '__main__':
    print('\n'.join(sheets(sys.argv[1], sys.argv[2], sys.argv[3])))
