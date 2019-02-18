import pandas as pd
import sys
from sleep_misc import load_dataset

TASK = int(sys.argv[1])
PATH_TO_VARIABLES = "./data/mesa/mesa-sleep-dataset-0.3.0.csv"

def get_time_interval(n):
    minutes = n / 2
    hours = minutes / 60
    rest_minutes = minutes - (hours * 60)
    rest_seconds = "30" if n%2 == 1 else "00"
    return "%02d:%02d:%s" % (hours, rest_minutes, rest_seconds)

DATASET_PATH = "hdf_task%d" % (TASK)
OUTPUT = "task%d_ml.csv" % (TASK)
SUMMARY_OUTPUT = "task%d_summary_ml.csv" % (TASK)

print("...Loading Task %d dataset into memory..." % (TASK))
dftrain, dftest, featnames = load_dataset(DATASET_PATH, useCache=True)
print("...Done...")

def print_stats(df, variables):
    df["linetime"] = pd.to_datetime(df["linetime"])

    print("Number of participants: %d" % (len(df.mesaid.unique())))
    print("Number of scored epochs: %d" % (df.shape[0]))

    longer_sleeptime = df.groupby("mesaid")["linetime"].apply(lambda x: len(x)).max()
    shorter_sleeptime = df.groupby("mesaid")["linetime"].apply(lambda x: len(x)).min()
    median_sleeptime = df.groupby("mesaid")["linetime"].apply(lambda x: len(x)).median()

    print("Longer Sleep period: %s" % (get_time_interval(longer_sleeptime)))
    print("Shorter Sleep period: %s" % (get_time_interval(shorter_sleeptime)))
    print("Median Sleep period: %s" % (get_time_interval(median_sleeptime)))

    ids = df["mesaid"].unique()
    varids = variables[variables["mesaid"].apply(lambda x: x in ids)]
    total = varids.shape[0]
    print("Number of Female: %d (%.2f%%)- Number of Male: %d (%.2f%%)" % ((varids["gender1"] == 0).sum(), 100.*(varids["gender1"] == 0).sum()/total,
                                                         (varids["gender1"] == 1).sum(), 100.*(varids["gender1"] == 1).sum()/total))
    print("White: %d (%.2f%%) - Chinese: %d (%.2f%%) - Black: %d (%.2f%%) - Hispanic: %d (%.2f%%)" % (
                (varids["race1c"] == 1).sum(), 100.*(varids["race1c"] == 1).sum()/total,
                (varids["race1c"] == 2).sum(), 100.*(varids["race1c"] == 2).sum()/total,
                (varids["race1c"] == 3).sum(), 100.*(varids["race1c"] == 3).sum()/total,
                (varids["race1c"] == 4).sum(), 100.*(varids["race1c"] == 4).sum()/total))
    print("Age: Mean: %.2f, Std: %.2f, Median: %.2f, Min: %d, Max:%d" % (varids["sleepage5c"].mean(), varids["sleepage5c"].std(),
            varids["sleepage5c"].median(), varids["sleepage5c"].min(), varids["sleepage5c"].max()))

variables = pd.read_csv(PATH_TO_VARIABLES)

print("\n...Stats for training set...")
print_stats(dftrain, variables)
print("\n...Stats for test set...")
print_stats(dftest, variables)
print("...Done...")
