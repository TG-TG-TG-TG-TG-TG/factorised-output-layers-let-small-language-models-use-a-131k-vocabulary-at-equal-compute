"""Round 12 evals: round11_eval.py's tasks (unchanged definitions; Part A found no evaluator bug, so E3 = round 11's
LAMBADA) on the round-12 checkpoints, written to runs/round12/eval_*.jsonl. Also E4 (round11_e4.evaluate) for D2's seed-1
bios models. The flat F128k head's eval chunk is 1,024 tokens (one fp32 logit chunk = 0.5 GB), as in training.

python round12_eval.py bpb|blimp|lambada|copy <labels>
python round12_eval.py e4 bioB_100000_s1,bioH_100000_eqF_s1
"""
import os
import sys

import round11_eval as E

E.OUT = os.path.join(E.HERE, "runs", "round12")
E.MODELS.update({
    "B384m_s0": ("B", "wsd_B384m_315M_s0"),
    "B384m_s1": ("B", "wsd_B384m_315M_s1"),
    "F128k384_84M": ("F128k384", "wsd_F128k384_161M_s0_branch84M"),
    "F128k384_161M": ("F128k384", "wsd_F128k384_161M_s0"),
    "H128k384_300M": ("H128k384", "wsd_H128k384_575M_s0_branch300M"),
    "H128k384_522M": ("H128k384", "wsd_H128k384_575M_s0_branch522M"),
    "H128k384_575M": ("H128k384", "wsd_H128k384_575M_s0"),
})

if __name__ == "__main__":
    task, labels = sys.argv[1], sys.argv[2].split(",")
    if task == "e4":
        import round11_e4 as E4
        E.MODELS.update({"bioB_100000_s1": ("bioB_100000", "wsd_bioB_100000_505M_s1"),
                         "bioH_100000_eqF_s1": ("bioH_100000", "wsd_bioH_100000_1200M_s1")})
        E4.EXPOSURES.update({"bioB_100000_s1": E4.EXPOSURES["bioB_100000"], "bioH_100000_eqF_s1": E4.EXPOSURES["bioH_100000_eqF"]})
        for lab in labels:
            E4.evaluate(lab)
    else:
        missing = [l for l in labels if not E.have(l)]
        assert not missing, f"no checkpoint for {missing}"
        print(f"[{task}] models: {labels}", flush=True)
        {"bpb": E.task_bpb, "blimp": E.task_blimp, "lambada": E.task_lambada, "copy": E.task_copy}[task](labels)
