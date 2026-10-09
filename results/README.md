# Results

`sweep5/` and `sweep6/`: evaluation output (`python -m chinukpipa.htr.evaluate ... --out`) of the recognizer runs summarized in
`chinukpipa/htr/README.md`. One file per configuration and word split (`base_v0` … `aug_v4`, `cal_v0` … `cal_v4`); `ens3_v0` is the
three-model combination on split `v0`. Each holds a
`summary` and, for every held-out word, its rule tokens (`ref`), the free reading (`greedy`) and the five best
word-list candidates (`top5`). The rule tokens are those in force when the runs were made (before the *z*
revision in `docs/rule_notes.md`). The model weights are not included.

`cv_rules_v2/`: the same kind of output for the runs with the revised spelling rules (the first table in
`chinukpipa/htr/README.md`), split `v0` … `v4` each: `base`, `synth3`, `bs128`, `synth3_bs128`, `synth3_gt1924`,
`synth3_bs128_gt1924` (one model, seed 0) and `base_ens3`, `synth3_ens3` (three models combined). These score the
472 held-out words of the 1892 and 1898 lists; the files ending in `_alltest` also score the held-out words of the
1924 *Rudiments* (`data/gt/rudiments1924_annotations.jsonl`), for the configurations named.

`hypotheses_rules_v2.json`: the rule tests of `docs/rule_notes.md` repeated with models trained on the revised
labels (summary per hypothesis, and per held-out occurrence the rule and alternative token sequences with their
losses under the model of that split).

`unseen_book_1924.json`: three models trained on all 1892 and 1898 words reading all 415 Chinook words of the 1924
*Rudiments*, which they had not seen (see the recognizer's README).

`creation_v0/`: the first running-text test page and its alignment with a printed Roman text (see its README).

`key1924_v0/`: the second running-text test, 823 words of the 1924 exercises read from correct word boxes (see its README).
