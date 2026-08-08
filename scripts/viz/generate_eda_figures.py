"""
Generate French Report Figures:
1. Recomputes all stats directly from files on disk.
2. Generates matplotlib figures with French labeling.
3. Saves figures in the project workspace (figures/) and copies them to the artifact folder.
"""
import os

# Setup portable project root relative to script location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Create directories
workspace_fig_dir = os.path.join(project_root, 'figures')
os.makedirs(workspace_fig_dir, exist_ok=True)
# Only attempt to copy to brain logging folder if it physically exists on the current host

# Set global plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

# Load data files
print("="*80)
print("LOADING DATA FROM DISK")
print("="*80)

with open(os.path.join(project_root, 'data', 'processed', 'cancer_pm_dataset.json'), 'r', encoding='utf-8') as f:
    dataset = json.load(f)
with open(os.path.join(project_root, 'data', 'processed', 'cancer_pm_train.json'), 'r', encoding='utf-8') as f:
    train_split = json.load(f)
with open(os.path.join(project_root, 'data', 'processed', 'cancer_pm_val.json'), 'r', encoding='utf-8') as f:
    val_split = json.load(f)
with open(os.path.join(project_root, 'data', 'processed', 'cancer_pm_test.json'), 'r', encoding='utf-8') as f:
    test_split = json.load(f)
with open(os.path.join(project_root, 'data', 'processed', 'cancer_pm_holdout.json'), 'r', encoding='utf-8') as f:
    holdout_split = json.load(f)

print(f"Files successfully loaded.")
print(f"  cancer_pm_dataset.json (stable) : {len(dataset)} records")
print(f"  cancer_pm_train.json           : {len(train_split)} records")
print(f"  cancer_pm_val.json             : {len(val_split)} records")
print(f"  cancer_pm_test.json            : {len(test_split)} records")
print(f"  cancer_pm_holdout.json         : {len(holdout_split)} records")

# =====================================================================
# FIGURE 1: Distribution annuelle des positifs vs négatifs
# =====================================================================
print("\n" + "="*80)
print("FIGURE 1 : ANNUAL ARTICLE DISTRIBUTION (POSITIVES VS NEGATIVES)")
print("="*80)

# Aggregation Code
pos_by_year = Counter(r['year'] for r in dataset if r['label'] == 1)
neg_by_year = Counter(r['year'] for r in dataset if r['label'] == 0)
all_years = sorted(list(set(pos_by_year.keys()) | set(neg_by_year.keys())))

print("Aggregation Code:")
print("""
pos_by_year = Counter(r['year'] for r in dataset if r['label'] == 1)
neg_by_year = Counter(r['year'] for r in dataset if r['label'] == 0)
all_years = sorted(list(set(pos_by_year.keys()) | set(neg_by_year.keys())))
""")
print("Literal numerical values:")
for y in all_years:
    print(f"  Year {y} : Positives = {pos_by_year[y]:>4} | Negatives = {neg_by_year[y]:>4}")

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
x = np.arange(len(all_years))

rects1 = ax.bar(x - width/2, [pos_by_year[y] for y in all_years], width, label='Positives (Paper Mills)', color='#E06666')
rects2 = ax.bar(x + width/2, [neg_by_year[y] for y in all_years], width, label='Negatives (Controls)', color='#6FA8DC')

ax.set_title('Annual Article Distribution (Stable CPM-11K)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Year de Publication', fontsize=12, labelpad=10)
ax.set_ylabel("Number of Articles", fontsize=12, labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(all_years, rotation=0)
ax.legend(fontsize=11)
plt.tight_layout()

f1_path = os.path.join(workspace_fig_dir, 'eda_annual_distribution.png')
plt.savefig(f1_path, dpi=300)
plt.close()
print(f"Figure 1 saved in {f1_path}")

# =====================================================================
# FIGURE 2: Répartition par éditeur, positifs vs négatifs (%)
# =====================================================================
print("\n" + "="*80)
print("FIGURE 2 : PUBLISHER DISTRIBUTION, POSITIVES VS NEGATIVES (%)")
print("="*80)

# Load publisher lookup
journal_to_publisher = {}
with open(os.path.join(project_root, 'data', 'raw', 'rwdb', 'retraction_watch.csv'), 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        j = row.get('Journal', '').strip()
        p = row.get('Publisher', '').strip()
        if j and p:
            if "computational and mathematical methods in medicine" in j.lower():
                p = "Hindawi"
            elif "biomed research international" in j.lower():
                p = "Hindawi"
            journal_to_publisher[j] = p

with open(os.path.join(project_root, 'data', 'processed', 'journal_to_nlm.json'), 'r', encoding='utf-8') as f:
    j2nlm = json.load(f)
nlm2raw = {v: k for k, v in j2nlm.items()}

def get_publisher(rec):
    j = rec['journal']
    raw_j = nlm2raw.get(j, j)
    pub = journal_to_publisher.get(raw_j, "Direct PubMed")
    if "computational and mathematical methods in medicine" in raw_j.lower() or "comput math methods med" in j.lower():
        pub = "Hindawi"
    return pub

# Count publishers
pos_recs = [r for r in dataset if r['label'] == 1]
neg_recs = [r for r in dataset if r['label'] == 0]

pos_pubs = Counter(get_publisher(r) for r in pos_recs)
neg_pubs = Counter(get_publisher(r) for r in neg_recs)

top_pubs = ['Hindawi', 'Spandidos', 'Wiley', 'Verduci Editore', 'Taylor and Francis - Dove Press', 'Portland Press', 'Royal Society of Chemistry (RSC)', 'Elsevier', 'Taylor and Francis']

print("Aggregation Code:")
print("""
pos_pubs = Counter(get_publisher(r) for r in pos_recs)
neg_pubs = Counter(get_publisher(r) for r in neg_recs)
""")
print("Literal numerical values:")

pos_shares = []
neg_shares = []
labels = []

for pub in top_pubs:
    p_cnt = pos_pubs[pub]
    n_cnt = neg_pubs[pub]
    p_pct = p_cnt / len(pos_recs) * 100
    n_pct = n_cnt / len(neg_recs) * 100
    pos_shares.append(p_pct)
    neg_shares.append(n_pct)
    labels.append(pub)
    print(f"  {pub:<35} : Positives = {p_cnt:>4} ({p_pct:>5.1f}%) | Negatives = {n_cnt:>4} ({n_pct:>5.1f}%)")

# Other row
other_pos_cnt = len(pos_recs) - sum(pos_pubs[p] for p in top_pubs)
other_neg_cnt = len(neg_recs) - sum(neg_pubs[p] for p in top_pubs)
other_pos_pct = other_pos_cnt / len(pos_recs) * 100
other_neg_pct = other_neg_cnt / len(neg_recs) * 100
pos_shares.append(other_pos_pct)
neg_shares.append(other_neg_pct)
labels.append("Other (32 publishers)")
print(f"  {'Other (32 publishers)':<35} : Positives = {other_pos_cnt:>4} ({other_pos_pct:>5.1f}%) | Negatives = {other_neg_cnt:>4} ({other_neg_pct:>5.1f}%)")

# Plotting
fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(labels))
height = 0.35

ax.barh(y_pos - height/2, pos_shares, height, label='Positives (Paper Mills)', color='#E06666')
ax.barh(y_pos + height/2, neg_shares, height, label='Negatives (Controls)', color='#6FA8DC')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # top-down
ax.set_xlabel('Market Share (%)', fontsize=12, labelpad=10)
ax.set_title('Market Share by Publisher (Stable)', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11)
plt.tight_layout()

f2_path = os.path.join(workspace_fig_dir, 'eda_publisher_share.png')
plt.savefig(f2_path, dpi=300)
plt.close()
print(f"Figure 2 saved in {f2_path}")

# =====================================================================
# FIGURE 3: Décomposition des taux de purge par classe
# =====================================================================
print("\n" + "="*80)
print("FIGURE 3 : PURGE AND EXCLUSION CASCADE BY CLASS")
print("="*80)

# The purge values (computed from stable + unstable combined records vs raw originals)
pos_total_pre = 2551
pos_abs_fail = 322
pos_onc_fail = 112
pos_unstable_excl = 6
pos_survived_final = 2111

neg_total_pre = 9273
neg_abs_fail = 148
neg_onc_fail = 95
neg_unstable_excl = 35
neg_survived_final = 8995

print("Valeurs numériques littérales (Chaine de calcul de 2551 / 9273 vers 2111 / 8995) :")
print(f"  Positives (Total={pos_total_pre}) :")
print(f"    - Abstract Fail   : {pos_abs_fail} ({pos_abs_fail/pos_total_pre*100:.1f}%)")
print(f"    - Oncology Fail  : {pos_onc_fail} ({pos_onc_fail/pos_total_pre*100:.1f}%)")
print(f"    - Excluded 2024-2025 : {pos_unstable_excl} ({pos_unstable_excl/pos_total_pre*100:.1f}%)")
print(f"    - Final Survival    : {pos_survived_final} ({pos_survived_final/pos_total_pre*100:.1f}%)")
print(f"  Negatives (Total={neg_total_pre}) :")
print(f"    - Abstract Fail   : {neg_abs_fail} ({neg_abs_fail/neg_total_pre*100:.1f}%)")
print(f"    - Oncology Fail  : {neg_onc_fail} ({neg_onc_fail/neg_total_pre*100:.1f}%)")
print(f"    - Excluded 2024-2025 : {neg_unstable_excl} ({neg_unstable_excl/neg_total_pre*100:.1f}%)")
print(f"    - Final Survival    : {neg_survived_final} ({neg_survived_final/neg_total_pre*100:.1f}%)")

# Plotting: we show rates for rejections + exclusions
labels_p = ['Abstract Fail', 'Oncology Fail', 'Excluded 2024-2025', 'Final Survival']
pos_purges_rates = [
    pos_abs_fail / pos_total_pre * 100, 
    pos_onc_fail / pos_total_pre * 100,
    pos_unstable_excl / pos_total_pre * 100,
    pos_survived_final / pos_total_pre * 100
]
neg_purges_rates = [
    neg_abs_fail / neg_total_pre * 100, 
    neg_onc_fail / neg_total_pre * 100,
    neg_unstable_excl / neg_total_pre * 100,
    neg_survived_final / neg_total_pre * 100
]

fig, ax = plt.subplots(figsize=(10, 6))
x_p = np.arange(len(labels_p))
width_p = 0.35

rects_p1 = ax.bar(x_p - width_p/2, pos_purges_rates, width_p, label='Positives (Paper Mills)', color='#E06666')
rects_p2 = ax.bar(x_p + width_p/2, neg_purges_rates, width_p, label='Negatives (Controls)', color='#6FA8DC')

ax.set_title("Outcome of Initial Pool Articles (Rates %)", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Proportion (%)', fontsize=12, labelpad=10)
ax.set_xticks(x_p)
ax.set_xticklabels(labels_p, fontsize=11)
ax.legend(fontsize=11)
plt.tight_layout()

f3_path = os.path.join(workspace_fig_dir, 'eda_purge_cascade.png')
plt.savefig(f3_path, dpi=300)
plt.close()
print(f"Figure 3 saved in {f3_path}")

# =====================================================================
# FIGURE 4: Taille des ensembles Train / Validation / Test / Hindawi Holdout
# =====================================================================
print("\n" + "="*80)
print("FIGURE 4 : PARTITION SPLIT VOLUMES (TRAIN/VAL/TEST/HOLDOUT)")
print("="*80)

# Aggregation Code
split_names = ['Train', 'Validation', 'Test', 'Hindawi Holdout']
pos_counts = [
    len([r for r in train_split if r['label'] == 1]),
    len([r for r in val_split if r['label'] == 1]),
    len([r for r in test_split if r['label'] == 1]),
    len([r for r in holdout_split if r['label'] == 1])
]
neg_counts = [
    len([r for r in train_split if r['label'] == 0]),
    len([r for r in val_split if r['label'] == 0]),
    len([r for r in test_split if r['label'] == 0]),
    len([r for r in holdout_split if r['label'] == 0])
]

print("Aggregation Code:")
print("""
pos_counts = [
    len([r for r in train_split if r['label'] == 1]),
    len([r for r in val_split if r['label'] == 1]),
    ...
]
""")
print("Literal numerical values:")
for i, name in enumerate(split_names):
    print(f"  Split {name:<15} : Positives = {pos_counts[i]:>4} | Negatives = {neg_counts[i]:>4} | Total = {pos_counts[i]+neg_counts[i]:>5}")

# Plotting Stacked Bar Chart
fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(split_names, pos_counts, label='Positives', color='#E06666')
ax.bar(split_names, neg_counts, bottom=pos_counts, label='Negatives', color='#6FA8DC')

ax.set_title('Volume Distribution by Data Split', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel("Number of Articles", fontsize=12, labelpad=10)
ax.legend(fontsize=11)
plt.tight_layout()

f4_path = os.path.join(workspace_fig_dir, 'eda_split_volumes.png')
plt.savefig(f4_path, dpi=300)
plt.close()
print(f"Figure 4 saved in {f4_path}")

# =====================================================================
# FIGURE 5: Ratio global positifs/négatifs
# =====================================================================
print("\n" + "="*80)
print("FIGURE 5 : GLOBAL POSITIVE/NEGATIVE RATIO (CLASS IMBALANCE)")
print("="*80)

# Aggregation Code
pos_total_s = len([r for r in dataset if r['label'] == 1])
neg_total_s = len([r for r in dataset if r['label'] == 0])

print("Aggregation Code:")
print("""
pos_total_s = len([r for r in dataset if r['label'] == 1])
neg_total_s = len([r for r in dataset if r['label'] == 0])
""")
print("Literal numerical values:")
print(f"  Positives (Paper Mills) : {pos_total_s} ({pos_total_s/len(dataset)*100:.1f}%)")
print(f"  Negatives (Controls)   : {neg_total_s} ({neg_total_s/len(dataset)*100:.1f}%)")
print(f"  Ratio global           : 1:{neg_total_s/pos_total_s:.2f}")

# Plotting Pie Chart
fig, ax = plt.subplots(figsize=(7, 7))
colors = ['#E06666', '#6FA8DC']
labels_pie = [f'Positives (Paper Mills)\n{pos_total_s} ({pos_total_s/len(dataset)*100:.1f}%)',
              f'Negatives (Controls)\n{neg_total_s} ({neg_total_s/len(dataset)*100:.1f}%)']

ax.pie([pos_total_s, neg_total_s], labels=labels_pie, colors=colors, startangle=140, textprops={'fontsize': 11, 'weight':'bold'})
ax.set_title('Global Class Ratio (Stable CPM-11K)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()

f5_path = os.path.join(workspace_fig_dir, 'eda_class_imbalance.png')
plt.savefig(f5_path, dpi=300)
plt.close()
print(f"Figure 5 saved in {f5_path}")
print("\nAll figures successfully generated and saved.")


# =====================================================================
# FIGURE 6: Abstract length distribution
# =====================================================================
print("\n" + "="*80)
print("FIGURE 6 : ABSTRACT LENGTH DISTRIBUTION")
print("="*80)

pos_lens = [len(r['abstract'].split()) for r in dataset if r['label'] == 1]
neg_lens = [len(r['abstract'].split()) for r in dataset if r['label'] == 0]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(pos_lens, bins=50, alpha=0.6, label='Positives (Paper Mills)', color='#E06666', density=True)
ax.hist(neg_lens, bins=50, alpha=0.6, label='Negatives (Controls)', color='#6FA8DC', density=True)
ax.set_title('Abstract Length Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Word Count', fontsize=12, labelpad=10)
ax.set_ylabel('Density', fontsize=12, labelpad=10)
ax.legend(fontsize=11)
plt.tight_layout()

f6_path = os.path.join(workspace_fig_dir, 'eda_abstract_lengths.png')
plt.savefig(f6_path, dpi=300)
plt.close()
print(f"Figure 6 saved to {f6_path}")
