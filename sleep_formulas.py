import pandas as pd
from glob import glob
import re
import sys
import os
from sleep_misc import load_dataset, apply_formulas_to_psgfile

TASK = int(sys.argv[1])

PATH_TO_FILES = "./datasets/task%d/*" % (TASK)

INPUTFILE="hdf_task%d" % (TASK)
SUMMARY_OUTPUT = "task%d_summary_formulas.csv" % (TASK)
OUTPUT = "task%d_formulas.csv" % (TASK)

print("...Loading Task %d dataset into memory..." % (TASK))
_, dftest, _ = load_dataset(INPUTFILE, useCache=True)
print("...Done...")

#Get unique test ids
uids_test = set(dftest.mesaid.unique())

def get_uid_from_filename(filename):
    #Find a particular uid from a filename
    return map(int, re.findall(r'\d+', filename))[0]

dfs = []
print("Found %d files in path %s" % (len(glob(PATH_TO_FILES)), PATH_TO_FILES))

for filename in glob(PATH_TO_FILES):
    uid = get_uid_from_filename(os.path.basename(filename))
    # Check uid present in the list of test uids
    if uid not in uids_test:
        continue
    # Process only test file
    print("Processing: %s" % (filename))
    # Run the formula based physical models (no randomization required) on 20% test samples
    dfs.append(apply_formulas_to_psgfile(filename))

formula_algs = ["time_based","sazonov", "sazonov2", "sadeh","cole","oakley", "kripke", "webster"]
p_formula_algs = ["p_sazonov","p_sazonov2","p_sadeh","p_cole","p_oakley","p_kripke","p_webster"]

dfs = pd.concat(dfs)
dfs["gt_sleep_block"] = dfs["gt_sleep_block"].astype(int)
dfs["gt"] = dfs["gt"].astype(int)
dfs["actValue"] = dfs["actValue"].fillna(0.0).astype(int)
#Select columns mesaid, linetime, activity value, groundtruth along with formula values and write to a csv file for inspection
dfs[["mesaid","linetime","actValue","gt","gt_sleep_block"] + formula_algs + p_formula_algs].to_csv(OUTPUT, index=False)

