EXP_GROUPS = {}
EXP_GROUPS["mnist"] = []

for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    EXP_GROUPS["mnist"] += [{"lr": lr, "dataset": "visual", "model": "linear"}]


EXP_GROUPS["syn"] = []

for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    EXP_GROUPS["syn"] += [{"lr": lr, "dataset": "syn", "model": "linear"}]
