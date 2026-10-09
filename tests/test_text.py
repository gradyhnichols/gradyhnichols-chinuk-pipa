"""Checks of the text pipeline (chinukpipa.text). CPU only; no data files, models or network.

Tests that need PyTorch or OpenCV are skipped where those are not installed.
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------------- batched reading


def _random_model(torch, seed: int, hidden: int):
    """A randomly initialised CRNN in eval mode. The BatchNorm layers get non-zero shifts and statistics, so that
    padded columns would not stay zero (and the test would notice) if the batched path let them leak in."""
    from chinukpipa.htr.model import CRNN
    torch.manual_seed(seed)
    m = CRNN(12, hidden=hidden)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.copy_(torch.randn_like(mod.running_mean) * 0.2)
            mod.running_var.copy_(1 + torch.rand_like(mod.running_var))
            mod.weight.data.copy_(1 + 0.2 * torch.randn_like(mod.weight))
            mod.bias.data.copy_(0.3 * torch.randn_like(mod.bias))
    return m.eval()


def _words(widths):
    """Fake normalized word images (uint8, 64 high, white background) of the given widths."""
    rng = np.random.default_rng(0)
    out = []
    for w in widths:
        ink = rng.random((64, w)) < 0.2
        out.append(np.where(ink, rng.integers(0, 120, (64, w)), 255).astype(np.uint8))
    return out


def test_batched_reading_equals_reading_each_word_alone():
    torch = pytest.importorskip("torch")
    from chinukpipa.text.readcrops import Ensemble
    models = [_random_model(torch, 1, 32), _random_model(torch, 2, 48)]
    ens = Ensemble.from_models(models)
    widths = [16, 17, 18, 19, 23, 40, 41, 57, 90, 101, 158]       # several are not divisible by 4
    arrs = _words(widths)
    with torch.no_grad():
        outs, T = ens.logps(arrs)
        for i, a in enumerate(arrs):
            x = torch.from_numpy(255 - a.astype(np.float32)).div(255.0)[None, None]
            assert int(T[i]) == widths[i] // 4
            for m, lp in zip(models, outs):
                alone = m(x)[:, 0]                                  # (T, C): the model's own forward pass
                assert alone.shape[0] == int(T[i])
                assert torch.allclose(lp[: int(T[i]), i], alone, atol=1e-4)
        # the result for a word does not depend on what else is in the batch
        sub = [arrs[9], arrs[2], arrs[5]]
        outs2, T2 = ens.logps(sub)
        for j, i in enumerate((9, 2, 5)):
            for lp, lp2 in zip(outs, outs2):
                assert torch.allclose(lp[: int(T[i]), i], lp2[: int(T2[j]), j], atol=1e-4)
        # the same holds for the confidence scores
        conf, conf_nb = ens.confidence(outs, T)
        for i, a in enumerate(arrs):
            c1, c1_nb = ens.confidence(*ens.logps([a]))
            assert conf[i] == pytest.approx(c1[0], abs=1e-5)
            assert conf_nb[i] == pytest.approx(c1_nb[0], abs=1e-5)


# ---------------------------------------------------------------------------------------- joining pieces


def test_compose_places_crops_at_their_boxes_and_keeps_the_darkest_pixel():
    from chinukpipa.text.remerge import compose
    pad = 4
    # crop A is cut at box (10, 20, 30, 40) +- pad: x 6..34, y 16..44. Crop B at (26, 22, 50, 44) +- pad:
    # x 22..54, y 18..48. The canvas starts at the top left of A and ends at the bottom right of B.
    a = np.full((28, 28), 255, np.uint8)
    b = np.full((30, 32), 255, np.uint8)
    a[2, 3] = 90                    # only in A: canvas (row 2, col 3)
    b[1, 2] = 30                    # only in B: B starts at canvas (row 2, col 16), so canvas (3, 18)
    a[10, 20], b[8, 4] = 50, 200    # the same canvas pixel (10, 20): the darker one, A's, wins
    a[12, 22], b[10, 6] = 220, 60   # the same canvas pixel (12, 22): B's wins
    canvas = compose([((10, 20, 30, 40), a), ((26, 22, 50, 44), b)], pad)
    assert canvas.dtype == np.uint8
    assert canvas.shape == (32, 48)          # x 6..54, y 16..48
    assert canvas[2, 3] == 90 and canvas[3, 18] == 30
    assert canvas[10, 20] == 50 and canvas[12, 22] == 60
    assert canvas[0, 40] == 255              # inside the canvas, outside both crops
    assert (canvas < 255).sum() == 4


def test_compose_clips_offsets_at_the_page_edge():
    from chinukpipa.text.remerge import compose
    pad = 4
    # box (2, 3, 20, 30): pad would reach x -2, y -1; the crop is cut at 0, so its offset is (0, 0)
    a = np.full((34, 24), 255, np.uint8)
    b = np.full((33, 28), 255, np.uint8)      # box (30, 5, 50, 30): offset (26, 1)
    b[0, 0] = 10
    a[33, 23] = 20
    canvas = compose([((2, 3, 20, 30), a), ((30, 5, 50, 30), b)], pad)
    assert canvas.shape == (34, 54)
    assert canvas[1, 26] == 10 and canvas[33, 23] == 20
    assert (canvas < 255).sum() == 2
    single = compose([((30, 5, 50, 30), b)], pad)       # one crop: the canvas is the crop itself
    assert np.array_equal(single, b)


# ---------------------------------------------------------------------------------------- stage A files


def _fake_crops():
    rng = np.random.default_rng(1)
    shapes = [(30, 41), (64, 17), (25, 90)]
    return [{"line": ln, "index": ix, "bbox": [10 * ix, 5 * ln, 10 * ix + w, 5 * ln + h],
             "pixels": rng.integers(0, 256, (h, w)).astype(np.uint8)}
            for (ln, ix), (h, w) in zip([(1, 1), (1, 2), (3, 1)], shapes)]


def test_segcrops_file_round_trips_through_readcrops(tmp_path):
    pytest.importorskip("cv2")                      # segcrops imports the page segmenter
    from chinukpipa.text.readcrops import load_npz
    from chinukpipa.text.segcrops import save_crops
    crops = _fake_crops()
    path = str(tmp_path / "book_15_words.npz")
    save_crops(path, crops)
    assert os.listdir(tmp_path) == ["book_15_words.npz"]          # no temporary file left behind
    assert set(np.load(path).files) == {"flat", "shapes", "boxes", "line", "index"}
    arrs, boxes, line, index = load_npz(path)
    assert len(arrs) == len(crops)
    for c, a in zip(crops, arrs):
        assert a.dtype == np.uint8 and np.array_equal(a, c["pixels"])
    assert boxes.tolist() == [c["bbox"] for c in crops]
    assert line.tolist() == [1, 1, 3] and index.tolist() == [1, 2, 1]


def test_segcrops_file_of_a_page_without_words(tmp_path):
    pytest.importorskip("cv2")
    from chinukpipa.text.readcrops import load_npz
    from chinukpipa.text.segcrops import save_crops
    path = str(tmp_path / "empty_words.npz")
    save_crops(path, [])
    arrs, boxes, line, index = load_npz(path)
    assert arrs == [] and boxes.shape == (0, 4) and len(line) == 0 and len(index) == 0


def test_dumpcrops_writes_the_crops_of_a_stage_a_file(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    from chinukpipa.text import dumpcrops
    from chinukpipa.text.segcrops import save_crops
    crops = _fake_crops()
    path = str(tmp_path / "p_words.npz")
    save_crops(path, crops)
    out = tmp_path / "png"
    monkeypatch.setattr(sys, "argv", ["dumpcrops", path, str(out)])
    dumpcrops.main()
    for c in crops:
        im = np.asarray(Image.open(out / f"L{c['line']:02d}_W{c['index']:02d}.png"))
        assert np.array_equal(im, c["pixels"])
    assert (out / "line_L01.png").exists() and (out / "line_L03.png").exists()


def test_word_crops_of_stage_a_and_read_page_are_the_same_cut():
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    from chinukpipa.text import read_page, segcrops
    rng = np.random.default_rng(2)
    labels = np.zeros((120, 200), np.int32)
    labels[40:60, 50:80] = 1          # a stroke of the word
    labels[30:34, 60:64] = 2          # a mark of the word
    labels[45:70, 84:100] = 3         # ink of the next word, inside the padded box of this one: must be whitened
    norm = rng.random((120, 200)).astype(np.float32)
    seg = {"stats": {"h_med": 40.0, "stroke_width": 7.0},     # pad 5 px, ink kept up to 3 px around the word
           "lines": [{"kind": "text", "number": 2, "words": [
               {"kind": "word", "comps": [1], "marks": [2], "bbox": [50, 30, 80, 60]},
               {"kind": "punct", "comps": [], "marks": [], "bbox": [90, 40, 95, 45]},
               {"kind": "word", "comps": [3], "marks": [], "bbox": [84, 45, 100, 70]}]},
               {"kind": "pagenum", "number": 0, "words": []}]}
    a = segcrops.word_crops(seg, norm, labels)
    b = read_page.word_crops(seg, norm, labels)
    assert [(c["line"], c["index"], c["bbox"]) for c in a] == [(2, 1, [50, 30, 80, 60]), (2, 3, [84, 45, 100, 70])]
    assert [(c["line"], c["index"], c["bbox"]) for c in b] == [(c["line"], c["index"], c["bbox"]) for c in a]
    for ca, cb in zip(a, b):
        assert np.array_equal(ca["pixels"], np.asarray(cb["image"]))
    # the first crop is cut at x 45..85, y 25..65. The page here is never fully white, so a pixel of 255 can only
    # come from the whitening: the word's stroke and mark and 3 px around the stroke are kept, the rest is white
    px = a[0]["pixels"]
    assert px.shape == (40, 40)
    assert (px[15:35, 5:35] < 255).all()            # the stroke
    assert (px[5:9, 15:19] < 255).all()             # the mark
    assert (px[20:30, 35:38] < 255).all()           # 1 to 3 px right of the stroke
    assert (px[20:30, 38:40] == 255).all()          # farther away, including the next word's ink at x = 84


# ---------------------------------------------------------------------------------------- scoring


ALIGN_HEADER = ["unit_kind", "main_box", "box_ids", "roman_tokens"]
ALIGN_ROWS = [["match", "1.1", "1.1", "A B"],
              ["match", "1.2", "1.2", "K A"],
              ["match", "1.3", "1.3", "M"],              # its box is missing from the stage-B file
              ["split", "1.4", "1.4+1.5", "S E L"],
              ["match", "brace", "brace", "T"]]          # no box at all


def _word(top, free, free_best, **extra):
    """A reading row: `top` = [(tokens, score)] of the word list, `free` = tokens, `free_best` = (tokens, score)."""
    w = {"top5": [{"tokens": t, "score": s} for t, s in top], "free_tokens": free,
         "free_best": None if free_best is None else {"tokens": free_best[0], "score": free_best[1]}}
    w.update(extra)
    return w


def test_score_alignment_strict_rule_for_read_and_merged_files(tmp_path):
    from chinukpipa.text import score_alignment as sa
    path = tmp_path / "alignment.tsv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(ALIGN_HEADER)
        wr.writerows(ALIGN_ROWS)
    al = sa.load_alignment(str(path))

    read = {"words": [
        _word([("A B", 4.0)], "A B", ("A B", 3.0), line=1, index=1),
        _word([("X", 9.0), ("Y", 9.5), ("K A", 9.9)], "K A", ("K A", 5.0), line=1, index=2),
        _word([("S E L", 2.0)], "S E L", ("S E L", 2.5), line=1, index=4)]}      # a box of a split word: not scored
    assert not sa.is_merged(read)
    r = sa.score(al, read)
    assert (r["top1"], r["top5"], r["free"], r["free_best"]) == (
        "1/5 = 20.0%", "2/5 = 40.0%", "2/5 = 40.0%", "2/5 = 40.0%")
    assert r["hybrid"]["0"] == "2/5 = 40.0%"           # the free reading is more than 0 better in both words
    assert r["hybrid"]["6"] == "1/5 = 20.0%"           # a margin of 6 keeps the list's answer for the second word
    assert r["hybrid"]["1000000000.0"] == "1/5 = 20.0%"
    assert list(r["hybrid"]) == [str(m) for m in sa.MARGINS]

    merged = {"joins": 3, "words": [
        _word([("A B", 4.0)], "A B", ("A B", 3.0), boxes=["1.1"]),
        _word([("K A M", 4.0)], "K A M", None, boxes=["1.2", "1.3"]),        # two single words wrongly joined
        _word([("S E L", 2.0)], "S E L", ("S E L", 2.5), boxes=["1.4", "1.5"])]}
    assert sa.is_merged(merged)
    m = sa.score(al, merged, margins=(0, 6))
    assert (m["top1"], m["top5"], m["free_best"]) == ("2/5 = 40.0%", "2/5 = 40.0%", "2/5 = 40.0%")
    assert m["right_by_unit_kind"] == {"match": 1, "split": 1}
    assert m["single_words_lost_to_wrong_joins"] == 2      # the words of 1.2 and 1.3
    assert m["joins"] == 3
    assert list(m["hybrid"]) == ["0", "6"]


def test_score_alignment_prints_one_block_per_file(tmp_path, capsys, monkeypatch):
    from chinukpipa.text import score_alignment as sa
    al = tmp_path / "a.tsv"
    al.write_text("\t".join(ALIGN_HEADER) + "\n" + "\t".join(ALIGN_ROWS[0]) + "\n", encoding="utf-8")
    rd = tmp_path / "p_read.json"
    rd.write_text(json.dumps({"words": [_word([("A B", 1.0)], "A B", None, line=1, index=1)]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["score_alignment", str(al), str(rd), "--margins", "0,2"])
    sa.main()
    out = capsys.readouterr().out
    assert out.startswith(str(rd) + " {") and '"top1": "1/1 = 100.0%"' in out
    assert sa.parse_margins("-2,0,1e9") == (-2, 0, 1e9)


# ---------------------------------------------------------------------------------------- word list script


def test_build_lj_wordlist_script(tmp_path):
    lex = tmp_path / "lex.tsv"
    lex.write_text("headword\tbest_conf\tn_sources\nkamooks\tA\t2\nwawa\tB\t3\nodd\tC\t1\n", encoding="utf-8")
    rows = [{"source": "S1", "language": "chn", "latin": "ta\tlla", "tokens_rule": "T A L A"},
            {"source": "S1", "language": "chn", "latin": "tala", "tokens_rule": "T A L A"},      # same tokens
            {"source": "S1", "language": "en", "latin": "dollar", "tokens_rule": "D"},          # not Chinook
            {"source": "S2", "language": "chn", "latin": "ikta", "tokens_rule": "I K T A"},
            {"source": "S2", "language": "chn", "latin": "no rule", "tokens_rule": None}]
    gt = tmp_path / "gt.jsonl"
    gt.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    out = tmp_path / "sub" / "list.tsv"
    run = [sys.executable, os.path.join(REPO, "scripts", "build_lj_wordlist.py"), "--lexicon", str(lex),
           "--gt", str(gt), "--out", str(out)]
    done = subprocess.run(run + ["--sources", "S1"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == ["headword\tbest_conf\ttokens\tsource",
                     "kamooks\tA\t\tlexicon", "wawa\tB\t\tlexicon",        # A/B headwords only, tokens empty
                     "ta lla\tA\tT A L A\tS1"]                              # tab in the spelling replaced
    bad = subprocess.run(run + ["--sources", "S3"], capture_output=True, text=True)
    assert bad.returncode != 0 and "S3" in bad.stderr


def test_gtrows_pack_and_score(tmp_path):
    """gtrows packs crops in reading order and scores readings keyed by page and position."""
    import json

    import numpy as np
    from PIL import Image

    from chinukpipa.text import gtrows
    from chinukpipa.text.readcrops import load_npz

    crops = tmp_path / "crops"
    crops.mkdir()
    rows = [
        {"id": "b_p3_r2", "page_index": 3, "seq": 2, "language": "chn", "tokens_rule": "K A", "bbox_shorthand": [0, 0, 5, 4]},
        {"id": "b_p3_r1", "page_index": 3, "seq": 1, "language": "chn", "tokens_rule": "M A", "bbox_shorthand": [0, 0, 6, 4]},
        {"id": "b_p4_r1", "page_index": 4, "seq": 1, "language": "en", "tokens_rule": None, "bbox_shorthand": [0, 0, 7, 4]},
    ]
    for k, r in enumerate(rows):
        Image.fromarray(np.full((4, 5 + k), 10 * k, np.uint8)).save(crops / f"{r['id']}_shorthand.png")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pages = tmp_path / "pages"
    gtrows.pack(str(rows_path), str(crops), str(pages), "bk")
    arrs, _, line, _ = load_npz(str(pages / "bk_3_words.npz"))
    assert [a.shape[1] for a in arrs] == [6, 5] and list(line) == [1, 2]      # seq order, not file order
    read = tmp_path / "read"
    read.mkdir()
    words = [{"line": 1, "index": 1, "free_tokens": "M A", "free_best": {"tokens": "M A", "score": 1.0},
              "top5": [{"tokens": "M A", "score": 1.0}]},
             {"line": 2, "index": 1, "free_tokens": "K", "free_best": None,
              "top5": [{"tokens": "K O", "score": 2.0}, {"tokens": "K A", "score": 3.0}]}]
    (read / "bk_3_read.json").write_text(json.dumps({"item": "bk", "leaf": 3, "words": words}), encoding="utf-8")
    lex = tmp_path / "lex.tsv"
    lex.write_text("headword\tbest_conf\ttokens\nma\tA\tM A\nko\tA\tK O\n", encoding="utf-8")
    res = gtrows.score(str(rows_path), str(pages), str(read), "bk", str(lex))
    assert res["words"] == 2 and res["list_first_choice"] == 0.5 and res["list_top5"] == 1.0
    assert res["in_list"] == 1 and res["in_list_first_choice"] == 1.0
    assert res["counts"]["edits"] == 1 and res["counts"]["ref_tokens"] == 4
