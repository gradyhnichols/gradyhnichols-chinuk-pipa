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

    pytest.importorskip("torch")
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


# ---------------------------------------------------------------------------------------- scoring with S.T. labels


@pytest.mark.parametrize("latin, want", [
    ("S.T.", "S T"), ("S.T", "S T"), ("s.t.", "S T"), (" S.T. ", "S T"),
    ("S.T. Papa", "S T _ P A P A"), ("S.T ya'ka", "S T _ E A K A"), ("Ya'ka S.T", "E A K A _ S T"),
    ("S.T. St Espli", "S T _ S T _ E S P L E"),           # "St" is a word the spelling rules read: S T
    ("S.T. Tanaz", "S T _ T A N A S"), ("S.T. Man", "S T _ M A N"),
    ("Papa", "P A P A"), ("kamooks", "K A M OO K S"), ("ka-ta", "K A _ T A"),
    ("pēl·telikom", None), ("Kin·jorj", None), ("S.T.·Papa", None),     # a raised dot: left unscored
    ("S.T. 2°", None), ("S.T. #", None),                                  # a word the rules cannot spell
    ("", None), ("   ", None), (None, None), (".", None),
])
def test_st_label_construction(latin, want):
    from chinukpipa.text.gtrows import st_label
    assert st_label(latin) == want


def test_st_label_agrees_with_the_spelling_rules_for_ordinary_words():
    from chinukpipa.text.gtrows import st_label
    from chinukpipa.translit import latin_to_tokens
    for word in ("kamooks", "alta", "kanawe", "chako", "kanamoxt", "khell", "kilapai", "wawa", "kloshe nanich"):
        assert st_label(word) == " ".join(latin_to_tokens(word))


def test_st_label_makes_a_label_for_the_s_t_rows_of_the_1924_exercises_and_not_for_the_dotted_ones():
    from chinukpipa.text.gtrows import load_rows, st_label
    rows = load_rows(os.path.join(REPO, "data", "gt", "rudiments1924_text_annotations.jsonl"))
    chn = [r for r in rows if r["language"] == "chn"]
    unlabelled = [r for r in chn if not (r.get("tokens_verified") or r.get("tokens_rule"))]
    made = {r["id"]: st_label(r["latin"]) for r in unlabelled}
    assert len(chn) == 860 and len(unlabelled) == 37
    assert sum(v is not None for v in made.values()) == 35                         # every row with S.T. ...
    assert sorted(r["latin"] for r in unlabelled if made[r["id"]] is None) == ["Kin·jorj", "pēl·telikom"]
    assert all("S T" in v for r in unlabelled if (v := made[r["id"]]))
    assert sum(1 for v in made.values() if v == "S T") == 20                       # 20 outlines of S.T. alone
    assert sum(1 for v in made.values() if v and " _ " in v) == 15                 # 15 with a second word


def _gt_scoring_fixture(tmp_path):
    rows = [{"id": "r1", "language": "chn", "latin": "ka", "tokens_rule": "K A"},
            {"id": "r2", "language": "chn", "latin": "ta", "tokens_rule": "T A"},
            {"id": "r3", "language": "chn", "latin": "S.T.", "tokens_rule": None},
            {"id": "r4", "language": "chn", "latin": "S.T. Papa", "tokens_rule": None},
            {"id": "r5", "language": "chn", "latin": "Ya'ka S.T", "tokens_rule": None},
            {"id": "r6", "language": "chn", "latin": "pēl·telikom", "tokens_rule": None},
            {"id": "r7", "language": "en", "latin": "fish", "tokens_rule": None},
            {"id": "r8", "language": "chn", "latin": "S.T. 2°", "tokens_rule": None}]
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "bk_rows.json").write_text(json.dumps({r["id"]: [1, n] for n, r in enumerate(rows, 1)}), encoding="utf-8")

    def w(n, free, top, free_best=None):
        return {"line": n, "index": 1, "free_tokens": free, "free_best": free_best,
                "top5": [{"tokens": t, "score": 1.0 + k} for k, t in enumerate(top)]}

    words = [w(1, "K A", ["K A", "T A"]), w(2, "T", ["K A", "T A"]),
             w(3, "S T", ["S T", "K A"], {"tokens": "S T", "score": 1.0}),
             w(4, "S T P A P A", ["S T _ P A P A", "S T"]),               # a two-word reading is the first choice
             w(5, "E A K A S T", ["S T", "K A"]),
             w(6, "", ["K A"]), w(7, "F", ["K A"]), w(8, "S T", ["S T"])]
    read = tmp_path / "read"
    read.mkdir()
    (read / "bk_1_read.json").write_text(json.dumps({"item": "bk", "leaf": 1, "words": words}), encoding="utf-8")
    main = tmp_path / "main.tsv"
    main.write_text("headword\tbest_conf\ttokens\nka\tA\tK A\nta\tA\tT A\npapa\tA\t\n", encoding="utf-8")
    return str(rows_path), str(pages), str(read), str(main)


def test_gtrows_score_with_and_without_st_labels(tmp_path):
    pytest.importorskip("torch")                         # lexicon_keys uses the recognizer's alphabet
    from chinukpipa.text import gtrows
    rows, pages, read, main = _gt_scoring_fixture(tmp_path)
    brief = os.path.join(REPO, "data", "lexicon", "brief_forms.tsv")

    plain = gtrows.score(rows, pages, read, "bk", main, extra_lexicons=[brief])
    assert "st_labels" not in plain and list(plain["counts"]) == [
        "words", "top1", "top5", "free", "free_best", "edits", "ref_tokens", "in_list", "top1_in_list",
        "top5_in_list", "chn_without_tokens", "no_reading"]
    assert plain["words"] == 2 and plain["counts"]["top1"] == 1 and plain["counts"]["top5"] == 2
    assert plain["counts"]["chn_without_tokens"] == 5 and plain["counts"]["ref_tokens"] == 4
    assert plain["in_list"] == 2

    st = gtrows.score(rows, pages, read, "bk", main, st_labels=True, extra_lexicons=[brief])
    c = st["counts"]
    assert (c["words"], c["top1"], c["top5"], c["free"], c["free_best"]) == (5, 3, 4, 2, 1)
    assert c["chn_without_tokens"] == 2 and c["ref_tokens"] == 4 + 2 + 7 + 7      # K A, T A | S T, S T _ P A P A, E A K A _ S T
    assert c["in_list"] == 3 and c["top1_in_list"] == 2                     # S T is in the list only with the extra list
    assert st["list_first_choice"] == 0.6 and st["list_top5"] == 0.8
    # the counts both ways: the rows that had label tokens, the rows labelled from their spelling, the rest
    assert st["st_labels"] == {
        "with_label_tokens": {"words": 2, "top1": 1, "top5": 2, "free": 1, "free_best": 0},
        "with_st_label": {"words": 3, "top1": 2, "top5": 2, "free": 1, "free_best": 1},
        "unscored": 2}
    # the totals are the two ways added up, and the first way is what the run without the option scored
    ways = st["st_labels"]
    for key in ("words", "top1", "top5", "free", "free_best"):
        assert c[key] == ways["with_label_tokens"][key] + ways["with_st_label"][key]
        assert plain["counts"][key] == ways["with_label_tokens"][key]
    # without the extra list, "S T" is not a candidate of the list that was read
    st2 = gtrows.score(rows, pages, read, "bk", main, st_labels=True)
    assert st2["counts"]["in_list"] == 2 and st2["counts"]["top1"] == 3


def test_gtrows_score_refuses_a_lexicon_that_differs_from_the_one_that_was_read(tmp_path):
    pytest.importorskip("torch")
    from chinukpipa.text import gtrows
    rows, pages, read, main = _gt_scoring_fixture(tmp_path)
    brief = os.path.join(REPO, "data", "lexicon", "brief_forms.tsv")
    path = os.path.join(read, "bk_1_read.json")
    d = json.load(open(path, encoding="utf-8"))
    d["n_candidates"] = 4                                # three from main.tsv and S.T. from the extra list
    json.dump(d, open(path, "w", encoding="utf-8"))
    gtrows.score(rows, pages, read, "bk", main, extra_lexicons=[brief])
    with pytest.raises(SystemExit):
        gtrows.score(rows, pages, read, "bk", main)       # the extra list is needed to count the candidates


def test_gtrows_score_records_the_pair_penalty_of_the_readings_and_refuses_a_mix(tmp_path):
    pytest.importorskip("torch")
    from chinukpipa.text import gtrows
    rows, pages, read, main = _gt_scoring_fixture(tmp_path)
    brief = os.path.join(REPO, "data", "lexicon", "brief_forms.tsv")
    path = os.path.join(read, "bk_1_read.json")
    d = json.load(open(path, encoding="utf-8"))
    d["n_candidates"] = 4
    assert "pair_penalty" not in gtrows.score(rows, pages, read, "bk", main, extra_lexicons=[brief])
    d["pair_penalty"] = 8.0                                # as readcrops writes it when the option is used
    json.dump(d, open(path, "w", encoding="utf-8"))
    res = gtrows.score(rows, pages, read, "bk", main, extra_lexicons=[brief])
    assert res["pair_penalty"] == 8.0 and res["words"] == 2
    # another page of the same book read without the option (readcrops skips pages that are already read)
    d2 = dict(d, leaf=2)
    del d2["pair_penalty"]
    json.dump(d2, open(os.path.join(read, "bk_2_read.json"), "w", encoding="utf-8"))
    with pytest.raises(SystemExit, match="different pair penalties"):
        gtrows.score(rows, pages, read, "bk", main, extra_lexicons=[brief])


def test_gtrows_command_line_takes_the_new_options(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from chinukpipa.text import gtrows
    rows, pages, read, main = _gt_scoring_fixture(tmp_path)
    brief = os.path.join(REPO, "data", "lexicon", "brief_forms.tsv")
    out = tmp_path / "score.json"
    monkeypatch.setattr(sys, "argv", ["gtrows", "score", rows, pages, read, "--book", "bk", "--lexicon", main,
                                      "--extra-lexicon", brief, "--st-labels", "--out", str(out)])
    gtrows.main()
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["words"] == 5 and res["counts"]["top1"] == 3 and res["st_labels"]["unscored"] == 2
    assert json.loads(capsys.readouterr().out)["words"] == 5
    monkeypatch.setattr(sys, "argv", ["gtrows", "score", rows, pages, read, "--book", "bk", "--lexicon", main,
                                      "--out", str(out)])
    gtrows.main()
    assert "st_labels" not in json.loads(out.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------- scoring: --merge-rule


MERGE_HEADER = ["unit_id", "unit_kind", "main_box", "box_ids", "roman_tokens"]
MERGE_ROWS = [["1", "match", "1.1", "1.1", "A B"],
              ["2", "split", "1.2", "1.2+1.3", "K A M"],
              ["3", "merge", "1.4", "1.4", "S A"], ["3", "merge", "1.4", "1.4", "T A"],                  # one box, two words
              ["4", "merge+split", "2.1", "2.1+2.2", "P A"], ["4", "merge+split", "2.1", "2.1+2.2", "K A"],
              ["4", "merge+split", "2.1", "2.1+2.2", "M A"],                                        # two boxes, three words
              ["5", "merge", "3.1", "3.1", "L A"], ["5", "merge", "3.1", "3.1", "N A"],            # read wrong
              ["6", "match", "3.2", "3.2", "D"]]                                                    # lost to a wrong join


def _merge_alignment(tmp_path, header=MERGE_HEADER, rows=MERGE_ROWS):
    path = tmp_path / "alignment.tsv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(header)
        wr.writerows([[r[MERGE_HEADER.index(h)] for h in header] for r in rows])
    return str(path)


def test_score_alignment_merge_rule_on_a_stage_c_file(tmp_path):
    from chinukpipa.text import score_alignment as sa
    al = sa.load_alignment(_merge_alignment(tmp_path))
    merged = {"joins": 1, "words": [
        _word([("A B", 4.0)], "A B", None, boxes=["1.1"]),
        _word([("K A M", 4.0)], "K A M", None, boxes=["1.2", "1.3"]),
        _word([("S A _ T A", 3.0), ("S A", 9.0)], "S A", ("S A _ T A", 3.5), boxes=["1.4"]),
        _word([("P A", 5.0), ("P A _ K A _ M A", 6.0)], "P A", ("P A _ K A _ M A", 2.0), boxes=["2.1", "2.2"]),
        _word([("X", 1.0)], "X", None, boxes=["3.1", "3.2"])]}
    strict = sa.score(al, merged)
    assert strict["top1"] == "2/10 = 20.0%" and strict["right_by_unit_kind"] == {"match": 1, "split": 1}
    assert strict["single_words_lost_to_wrong_joins"] == 1
    assert sa.score(al, merged, merge_rule=False) == strict

    r = sa.score(al, merged, margins=(0, 1e9), merge_rule=True)
    # unit 3 (one box, first choice S A _ T A): its two words are right; unit 4 (first choice P A): wrong, although
    # its second choice and its free_best are right; unit 5 has no output word of its own (its box was joined to the
    # next one), unit 6 is lost to that join
    assert r["top1"] == "4/10 = 40.0%"
    assert r["top5"] == "7/10 = 70.0%"                # the two single words, unit 3 (2) and unit 4 (3: second choice)
    assert r["free_best"] == "5/10 = 50.0%"           # unit 3 (2) and unit 4 (3)
    assert r["right_by_unit_kind"] == {"match": 1, "split": 1, "merge": 2, "merge+split": 0}
    assert r["single_words_lost_to_wrong_joins"] == 1 and r["joins"] == 1
    # hybrid: free_best where its loss plus the margin is lower than the first choice's. Unit 4 (2.0 vs 5.0) is right
    # with the free reading at margin 0 and wrong with the first choice (margin 1e9); unit 3 is right either way
    assert r["hybrid"] == {"0": "7/10 = 70.0%", "1000000000.0": "4/10 = 40.0%"}


def test_score_alignment_merge_rule_counts_each_word_of_a_covered_unit(tmp_path):
    from chinukpipa.text import score_alignment as sa
    al = sa.load_alignment(_merge_alignment(tmp_path))
    merged = {"joins": 0, "words": [
        _word([("S A _ T A", 3.0)], "S A _ T A", ("S A _ T A", 3.0), boxes=["1.4"]),
        _word([("P A _ K A _ M A", 3.0)], "P A _ K A _ M A", ("P A _ K A _ M A", 3.0), boxes=["2.1", "2.2"]),
        _word([("L A _ N A", 3.0)], "L A _ N A", ("L A _ N A", 3.0), boxes=["3.1"]),
        _word([("D", 3.0)], "D", ("D", 3.0), boxes=["3.2"])]}
    # strict: no unit with two words counts; the match unit D does
    assert sa.score(al, merged)["top1"] == "1/10 = 10.0%"
    r = sa.score(al, merged, merge_rule=True)
    assert r["top1"] == "8/10 = 80.0%"                 # 2 + 3 + 2 words of the three units, + D
    assert r["top5"] == "8/10 = 80.0%" and r["free_best"] == "8/10 = 80.0%"
    assert r["right_by_unit_kind"] == {"match": 1, "merge": 4, "merge+split": 3}
    assert all(v == "8/10 = 80.0%" for v in r["hybrid"].values())


def test_score_alignment_merge_rule_requires_the_exact_boxes_and_the_exact_joined_tokens(tmp_path):
    from chinukpipa.text import score_alignment as sa
    al = sa.load_alignment(_merge_alignment(tmp_path))
    for tokens, boxes in (("S A T A", ["1.4"]),                 # no word-space between the words
                          ("T A _ S A", ["1.4"]),               # the words in the wrong order
                          ("S A _ T A", ["1.4", "1.5"]),        # an output word that covers more than the unit
                          ("S A", ["1.4"])):                    # only one of the two words
        merged = {"joins": 0, "words": [_word([(tokens, 1.0)], tokens, (tokens, 1.0), boxes=boxes)]}
        assert sa.score(al, merged, merge_rule=True)["top1"] == "0/10 = 0.0%", (tokens, boxes)
    ok = {"joins": 0, "words": [_word([("S A _ T A", 1.0)], "", None, boxes=["1.4"])]}
    assert sa.score(al, ok, merge_rule=True)["top1"] == "2/10 = 20.0%"
    # the unit's rows need not carry a unit_id: they are found by their boxes
    al2 = sa.load_alignment(_merge_alignment(tmp_path, header=["unit_kind", "main_box", "box_ids", "roman_tokens"]))
    assert sa.score(al2, ok, merge_rule=True)["top1"] == "2/10 = 20.0%"


def test_score_alignment_merge_rule_on_a_stage_b_file(tmp_path):
    from chinukpipa.text import score_alignment as sa
    al = sa.load_alignment(_merge_alignment(tmp_path))
    read = {"words": [
        _word([("A B", 4.0)], "A B", None, line=1, index=1),
        _word([("S A _ T A", 3.0)], "S A _ T A", None, line=1, index=4),     # the box of unit 3 reads as two words
        _word([("P A _ K A _ M A", 3.0)], "P A _ K A _ M A", None, line=2, index=1)]}   # one box of unit 4: no
    assert not sa.is_merged(read)
    strict = sa.score(al, read)
    assert strict["top1"] == "1/10 = 10.0%" and strict["free"] == "1/10 = 10.0%"
    r = sa.score(al, read, merge_rule=True)
    assert r["top1"] == "3/10 = 30.0%" and r["free"] == "3/10 = 30.0%"       # + the two words of the one-box unit


def test_score_alignment_merge_rule_command_line(tmp_path, capsys, monkeypatch):
    from chinukpipa.text import score_alignment as sa
    al = _merge_alignment(tmp_path)
    rd = tmp_path / "p_merged.json"
    rd.write_text(json.dumps({"joins": 0, "words": [
        _word([("S A _ T A", 1.0)], "", None, boxes=["1.4"])]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["score_alignment", al, str(rd)])
    sa.main()
    assert '"top1": "0/10 = 0.0%"' in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["score_alignment", al, str(rd), "--merge-rule", "--margins", "0"])
    sa.main()
    assert '"top1": "2/10 = 20.0%"' in capsys.readouterr().out


def test_score_alignment_without_the_option_is_what_it_was_on_the_archived_creation_readings():
    """The first running-text test page (results/creation_v0): the strict scores from before --merge-rule."""
    from chinukpipa.text import score_alignment as sa
    al = sa.load_alignment(os.path.join(REPO, "results", "creation_v0", "alignment.tsv"))
    d = json.load(open(os.path.join(REPO, "results", "creation_v0", "readings_final6_stageC.json"), encoding="utf-8"))
    r = sa.score(al, d)
    assert (r["top1"], r["top5"], r["free_best"]) == ("153/208 = 73.6%", "157/208 = 75.5%", "150/208 = 72.1%")
    assert r["right_by_unit_kind"] == {"split": 17, "match": 136}
    assert r["single_words_lost_to_wrong_joins"] == 5 and r["joins"] == 112
    assert r["hybrid"] == {"-2": "150/208 = 72.1%", "0": "150/208 = 72.1%", "1": "151/208 = 72.6%",
                           "2": "152/208 = 73.1%", "3": "154/208 = 74.0%", "4": "155/208 = 74.5%",
                           "6": "157/208 = 75.5%", "8": "158/208 = 76.0%", "1000000000.0": "153/208 = 73.6%"}
    # these readings have no two-word first choice, so the option finds nothing more
    m = sa.score(al, d, merge_rule=True)
    assert m["top1"] == r["top1"] and m["top5"] == r["top5"] and m["free_best"] == r["free_best"]
    assert m["right_by_unit_kind"]["match"] == 136 and m["right_by_unit_kind"]["split"] == 17


def _gaps(rng, n_small, small, n_word, word, n_wide=0, wide=300.0):
    """Synthetic nearest-ink gaps of one page: pen lifts inside words, spaces between words, a few very wide gaps."""
    g = list(rng.lognormal(np.log(small), 0.35, n_small)) + list(rng.lognormal(np.log(word), 0.25, n_word))
    return g + list(rng.lognormal(np.log(wide), 0.1, n_wide))


def test_gap_threshold_uses_the_valley_between_pen_lifts_and_word_spaces():
    pytest.importorskip("cv2")
    from chinukpipa.text.segment import gap_threshold
    rng = np.random.default_rng(0)
    thr, st = gap_threshold(_gaps(rng, 600, 5.0, 500, 25.0, n_wide=15), h_med=45.0)
    assert st["method"] == "kde valley" and 8 < thr < 20
    assert st["n_above"] >= 0.25 * st["n_gaps"]


def test_gap_threshold_rejects_a_valley_with_almost_no_gaps_above_it():
    """A broad word-space bump that is no peak of its own, and a few very wide gaps (between columns, around a
    picture): the strongest pair of modes is then pen lifts vs the wide gaps, and its valley would join whole lines.
    Fewer than 25% of the gaps lie above it, so Otsu's threshold is used instead (cihm_14939 leaf 9: 19 of 1,480)."""
    pytest.importorskip("cv2")
    from chinukpipa.text.segment import MIN_SHARE_ABOVE, gap_threshold
    rng = np.random.default_rng(1)
    lifts_and_spaces = list(rng.lognormal(np.log(8.0), 0.7, 1300))   # one broad bump: no word-space peak of its own
    wide = list(rng.lognormal(np.log(320.0), 0.05, 19))
    thr, st = gap_threshold(lifts_and_spaces + wide, h_med=39.0)
    rejected = st["rejected_kde_valley"]                          # the valley that was found ...
    assert rejected["n_above"] < MIN_SHARE_ABOVE * st["n_gaps"] and rejected["valley_px"] > 100
    assert st["method"].startswith("otsu (kde valley rejected")   # ... is rejected, and Otsu's threshold used
    assert thr == pytest.approx(st["otsu_px"], abs=0.05) and thr < 39.0
    assert "mode_large_px" not in st     # stage C falls back to 2.2 x the threshold for its join limit
