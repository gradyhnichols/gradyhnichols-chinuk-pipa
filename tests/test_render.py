import glob

import pytest

fonts = sorted(glob.glob("fonts/*.otf") + glob.glob("fonts/*.ttf"))


@pytest.mark.skipif(not fonts, reason="no Duployan font in fonts/")
def test_render_produces_ink():
    pytest.importorskip("uharfbuzz")
    from chinukpipa.render import Renderer
    from chinukpipa.translit import latin_to_tokens, tokens_to_unicode
    im = Renderer(fonts[0]).render(tokens_to_unicode(latin_to_tokens("kamooks")), size=48)
    assert im.width > 10 and im.height > 10
    assert min(im.getdata()) < 128   # some dark pixels


@pytest.mark.skipif(not fonts, reason="no Duployan font in fonts/")
def test_dotted_circle_detected_for_invalid_sequence():
    pytest.importorskip("uharfbuzz")
    from chinukpipa.render import Renderer
    from chinukpipa.translit import tokens_to_unicode
    r = Renderer(fonts[0])
    assert r.has_dotted_circle(tokens_to_unicode(["P", "OO", "L", "E", "E"]))
    assert not r.has_dotted_circle(tokens_to_unicode(["K", "A", "M", "OO", "K", "S"]))
