import pandas as pd
from glob import glob
import re
import sys
from sleep_misc import load_dataset, grid_search, load_mesa_PSG
from sleep_misc import time_based
from sleep_misc import oakley
from sleep_eval import eval_acc

TASK = int(sys.argv[1])
PATH_TO_FILES = "./datasets/task%d/*" % (TASK)

INPUTFILE="hdf_task%d" % (TASK)
SUMMARY_OUTPUT = "task%d_summary_formulas.csv" % (TASK)
OUTPUT = "task%d_formulas.csv" % (TASK)

print("...Loading Task %d dataset into memory..." % (TASK))
dftrain, _, _ = load_dataset(INPUTFILE, useCache=True)
print("...Done...")

uids_train = set(dftrain.mesaid.unique())

def get_uid_from_filename(filename):
    return map(int, re.findall(r'\d+', filename))[1]

dfs = []
for filename in glob(PATH_TO_FILES):

    uid = get_uid_from_filename(filename)
    if uid not in uids_train:
        continue

    print("Processing: %s" % (filename))
    df = load_mesa_PSG(filename)
    dfs.append(df)

df = pd.concat(dfs)
df.reset_index(inplace=True,drop=True)

# Time Based:
parameters = {"min_sleep":range(5,61,5), "min_awaken":range(5,60,5)}
results = grid_search(df, time_based, parameters, eval_acc)

# Oakley:
parameters = {"threshold":range(5,300,5)}
results = grid_search(df, oakley, parameters, eval_acc)
pd.DataFrame(results).to_csv("oakley_parameters.csv", index=False)

