# Knowledge Distillation: Does it hurt classes unevenly?

Starter code for the project. Five files:

- `data.py` — CIFAR-100 and CIFAR-100-LT loaders, plus a robust fine→superclass mapping read directly from the dataset's raw pickle (no hardcoded lookup table to get wrong).
- `models.py` — teacher (ResNet-50/34, ImageNet-pretrained, CIFAR-adapted stem) and student (ResNet-18 or a deliberately small custom CNN).
- `losses.py` — standard Hinton KD loss, plus an optional class-balanced variant (effective-number reweighting) for the mitigation experiment.
- `train.py` — one script, four modes: `teacher`, `student_plain`, `student_kd`, `student_kd_cb`.
- `analyze.py` — the diagnostic script: per-class accuracy deltas, flip analysis, frequency/confusability correlation, ECE + reliability diagrams.

## Install

```bash
pip install torch torchvision numpy scipy matplotlib
```

## Recommended run sequence (maps to your days 3–7)

```bash
# 1. Fine-tune teacher (balanced CIFAR-100 — you generally want the teacher
#    itself to be strong/unbiased so any disparate impact you find is
#    attributable to distillation, not to a broken teacher)
python train.py --mode teacher --arch resnet50 --epochs 30 \
    --save ckpts/teacher.pt

# 2. Plain student baseline, no distillation
python train.py --mode student_plain --student resnet18 --epochs 40 \
    --save ckpts/student_plain.pt

# 3. Standard KD student (Hinton et al., T=4, alpha=0.5 as agreed)
python train.py --mode student_kd --student resnet18 --epochs 40 \
    --teacher_ckpt ckpts/teacher.pt --teacher_arch resnet50 \
    --T 4.0 --alpha 0.5 --save ckpts/student_kd.pt

# 4. Diagnostic analysis (headline plots + summary.json)
python analyze.py \
    --teacher_ckpt ckpts/teacher.pt --teacher_arch resnet50 \
    --student_plain_ckpt ckpts/student_plain.pt \
    --student_kd_ckpt ckpts/student_kd.pt \
    --student resnet18 \
    --out_dir results/
```

If you also want the class-frequency correlate (not just confusability),
train steps 2–3 on CIFAR-100-LT instead by adding `--lt --imb_factor 0.01`,
and pass the same flags to `analyze.py` so it can compute `class_counts`.
Note the analysis always *evaluates* on the standard balanced test set
either way — only the training distribution changes — so per-class
accuracy stays comparable.

## Optional mitigation (days 8–9)

```bash
python train.py --mode student_kd_cb --student resnet18 --epochs 40 \
    --teacher_ckpt ckpts/teacher.pt --teacher_arch resnet50 \
    --lt --imb_factor 0.01 --T 4.0 --alpha 0.5 \
    --save ckpts/student_kd_cb.pt

python analyze.py \
    --teacher_ckpt ckpts/teacher.pt --teacher_arch resnet50 \
    --student_plain_ckpt ckpts/student_plain.pt \
    --student_kd_ckpt ckpts/student_kd.pt \
    --student_kd_cb_ckpt ckpts/student_kd_cb.pt \
    --student resnet18 --lt --imb_factor 0.01 \
    --out_dir results_with_mitigation/
```

## What `analyze.py` gives you (headline deliverables)

- `per_class_delta.png` — the headline plot: sorted bar chart of
  (KD student accuracy − plain student accuracy) per class, red/green
  colored. This directly answers "does KD hurt classes uniformly?"
- `summary.json` — overall accuracies, mean/std of the delta, counts of
  classes helped vs hurt, worst-hurt and most-helped class names, and
  which classes lost the most individual examples (regressions).
- `regressed_indices.npy` / `improved_indices.npy` — test-set indices
  that flipped correct→incorrect (or the reverse) under distillation, so
  you can pull actual images for the presentation's qualitative slide.
- `delta_vs_confusability.png` + Pearson/Spearman stats — tests whether
  classes visually/semantically confusable with their CIFAR-100 superclass
  siblings are disproportionately hurt.
- `delta_vs_frequency.png` (only with `--lt`) — tests the class-frequency
  hypothesis.
- `reliability_*.png` + `ece_*` in `summary.json` — calibration comparison
  across teacher / plain student / KD student (and the mitigation variant
  if provided).

## Notes / things you'll likely want to tweak

- **Epoch counts** above are placeholders — on a shared cluster, start with
  a short run (e.g. 5–10 epochs) end-to-end to catch bugs before committing
  GPU-hours to the full 30–40 epoch runs.
- **Capacity gap**: ResNet50→ResNet18 is a fairly mild teacher/student gap.
  If your early results show KD barely changing anything, swap `--student
  small_cnn` for a starker gap — more likely to produce a visible, analyzable
  effect within your timeline.
- **`num_workers`** in `data.py` defaults to 4; adjust to your cluster's
  CPU allocation.
- The `--lt` imbalance profile follows the standard exponential-decay
  construction from the CIFAR-LT literature (Cao et al. 2019 / Cui et al.
  2019) — imb_factor=0.01 means a 100:1 ratio between the largest and
  smallest class, a common default worth citing in your report.
