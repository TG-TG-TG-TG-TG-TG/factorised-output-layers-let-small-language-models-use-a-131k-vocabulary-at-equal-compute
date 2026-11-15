"""Round 13 evals (2026-11-12): round11_eval.py's byte-aligned BPB (task_bpb, unchanged definition) on the experiment-3
adaptive-softmax checkpoints, written to runs/round13/eval_bpb.jsonl. 131k: the BPB of record (as F128k / H128k in
round 12). 50k: a cross-check only (the BPB of record is val_loss_full x BPN, round 11's gate).

python round13_eval.py bpb AS128k384_620M,AS128k384_323M[,AS384_624M_s0] [skipmissing]
(skipmissing, 2026-11-13: evaluate the labels that have a checkpoint and list the others, instead of stopping)
"""
import os
import sys

import round11_eval as E

E.OUT = os.path.join(E.HERE, "runs", "round13")
E.MODELS.update({
    "AS128k384_323M": ("AS128k384s", "wsd_AS128k384_620M_s0_branch323M"),
    "AS128k384_620M": ("AS128k384s", "wsd_AS128k384_620M_s0"),
    "AS384_624M_s0": ("AS384", "wsd_AS384_624M_s0"),
    "AS384_624M_s1": ("AS384", "wsd_AS384_624M_s1"),
    # experiment 4 (2026-11-13): 131k seed 1 and the second adaptive-softmax configuration
    "H128k384_522M_s1": ("H128k384", "wsd_H128k384_575M_s1_branch522M"),
    "H128k384_300M_s1": ("H128k384", "wsd_H128k384_575M_s1_branch300M"),
    "AS128k384_620M_s1": ("AS128k384s", "wsd_AS128k384_620M_s1"),
    "AS128k384_323M_s1": ("AS128k384s", "wsd_AS128k384_620M_s1_branch323M"),
    "AS128k384c49_612M": ("AS128k384c49", "wsd_AS128k384c49_612M_s0"),
    "AS128k384c49_319M": ("AS128k384c49", "wsd_AS128k384c49_612M_s0_branch319M"),
    "AS128k384c49_612M_s1": ("AS128k384c49", "wsd_AS128k384c49_612M_s1"),
    "AS128k384c49_319M_s1": ("AS128k384c49", "wsd_AS128k384c49_612M_s1_branch319M"),
    # experiment 5 (2026-11-13): setup 2 with full-width tail bands (no projection, untied)
    "AS128k384c49w_540M": ("AS128k384c49w", "wsd_AS128k384c49w_540M_s0"),
    "AS128k384c49w_282M": ("AS128k384c49w", "wsd_AS128k384c49w_540M_s0_branch282M"),
    "AS128k384c49w_540M_s1": ("AS128k384c49w", "wsd_AS128k384c49w_540M_s1"),
    "AS128k384c49w_282M_s1": ("AS128k384c49w", "wsd_AS128k384c49w_540M_s1_branch282M"),
})

if __name__ == "__main__":
    task, labels = sys.argv[1], sys.argv[2].split(",")
    missing = [l for l in labels if not E.have(l)]
    if len(sys.argv) > 3 and sys.argv[3] == "skipmissing":
        print(f"[{task}] no checkpoint, skipped: {missing}", flush=True)
        labels = [l for l in labels if l not in missing]
    else:
        assert not missing, f"no checkpoint for {missing}"
    print(f"[{task}] models: {labels}", flush=True)
    {"bpb": E.task_bpb}[task](labels)
