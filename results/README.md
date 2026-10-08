# Results

`sweep5/` and `sweep6/`: evaluation output (`python -m chinukpipa.htr.evaluate ... --out`) of the recognizer runs summarized in
`chinukpipa/htr/README.md`. One file per configuration and word split (`base_v0` … `aug_v4`, `cal_v0` … `cal_v4`); `ens3_v0` is the
three-model combination on split `v0`. Each holds a
`summary` and, for every held-out word, its rule tokens (`ref`), the free reading (`greedy`) and the five best
word-list candidates (`top5`). The rule tokens are those in force when the runs were made (before the *z*
revision in `docs/rule_notes.md`). The model weights are not included.
