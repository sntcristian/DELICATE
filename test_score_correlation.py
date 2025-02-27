import pandas as pd
from scipy.stats import pointbiserialr


df_elite = pd.read_csv("DZ_results/elite_ed_no_stddev.csv")
df_blink = pd.read_csv("../ENEIDE/results/DZ/mgenre_impresso_ed/output.csv")
df_gt = pd.read_csv("../ENEIDE/DZ/v0.1/annotations_test.csv")


common_fields = ["doc_id", "surface", "start_pos", "end_pos", "type"]

df_elite_merged = pd.merge(
    df_elite,
    df_gt,
    on=common_fields,
    how="inner",
    suffixes=("", "_gt")
)

df_blink_merged = pd.merge(
    df_blink,
    df_gt,
    on=common_fields,
    how="inner",
    suffixes=("", "_gt")
)


df_elite_merged["correct"] = (
    df_elite_merged["identifier"] == df_elite_merged["identifier_gt"]
).astype(int)

df_blink_merged["correct"] = (
    df_blink_merged["identifier"] == df_blink_merged["identifier_gt"]
).astype(int)


elite_corr, elite_pval = pointbiserialr(df_elite_merged["correct"], df_elite_merged["score"])
MGENRE_corr, MGENRE_pval = pointbiserialr(df_blink_merged["correct"], df_blink_merged["score"])

print(f"ELITE - Point-biserial correlation: {elite_corr:.4f}, p-value: {elite_pval:.4e}")
print(f"MGENRE impresso - Point-biserial correlation: {MGENRE_corr:.4f}, p-value: {MGENRE_pval:.4e}")



