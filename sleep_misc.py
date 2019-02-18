import pandas as pd
import numpy as np
from glob import glob
import os
from itertools import product

def load_dataset(path, useCache=False, saveCache=False, cacheName="hdf", ground_truth="stage"):

    # Load cached dataset
    if useCache:
        store = pd.HDFStore(path)
        dftrain = store["train"]
        dftest = store["test"]
        featnames = list(store["featnames"].values)
        store.close()
        return dftrain, dftest, featnames

    # Or....load the dataset from scratch
    tmp = []

    for filename in glob(os.path.join(path, "*"))[:]:
        print(filename)
        dftmp = load_mesa_PSG(filename, ground_truth)

        # creates a gt_block
        gtTrue = dftmp[dftmp["gt"] == True]
        if gtTrue.empty:
            print("Ignoring file %s" % (filename))
            continue
        start_block = dftmp.index.get_loc(gtTrue.index[0])
        end_block =  dftmp.index.get_loc(gtTrue.index[-1])
        dftmp["gt_sleep_block"] = make_one_block(dftmp["gt"], start_block, end_block)

        featnames = get_features(dftmp)
        tmp.append(dftmp)

    wholedf = pd.concat(tmp)
    del tmp
    wholedf.reset_index(inplace=True, drop=True)

    # Generates a binary version of the interval col
    wholedf["binterval"] = wholedf["interval"].replace("ACTIVE", 0).replace("REST",1).replace("REST-S", 1)

    # Splits uids into training and test sets.
    test_proportion = 0.2

    uids = wholedf.mesaid.unique()
    np.random.seed(42)
    np.random.shuffle(uids)
    test_position = int(uids.shape[0] * test_proportion)

    uids_test, uids_train = uids[:test_position], uids[test_position:]

    # Splits dataset into training and test sets.
    train_idx = wholedf[wholedf["mesaid"].apply(lambda x: x in uids_train)].index
    dftrain = wholedf.iloc[train_idx].copy()

    test_idx = wholedf[wholedf["mesaid"].apply(lambda x: x in uids_test)].index
    dftest = wholedf.iloc[test_idx].copy()

    if saveCache:
        store = pd.HDFStore(cacheName)
        store["train"] = dftrain
        store["test"] = dftest
        store["featnames"] = pd.Series(featnames)
        store.close()

    return dftrain, dftest, featnames

def rescore_models(df, models, tl_min_sleep=10, tl_min_awake=20):
    """
      Increment a model with additional data from rescoring methods (e.g., Tudor Locke, Webster Rescoring Rules).
      Directly applies rules to the input DF and returns the list of new cols created by this method.
    """


    all_models = []
    for model in models:
        print("Creating new cols for alg %s..." % (model))
        df[model] = df[model].astype(np.bool)
        all_models.append(model)

        df["tl_" + model] = tudor_locke(df[model], min_minutes_sleep=tl_min_sleep, min_minutes_awake=tl_min_awake)
        df["resc_" + model] = webster_rescoring_rules(df[model])
        df["resc_tl_" + model] = webster_rescoring_rules(df["tl_" + model])
        df["tl_resc_" + model] = tudor_locke(df["tl_" + model], min_minutes_sleep=tl_min_sleep, min_minutes_awake=tl_min_awake)
        df[model + "_max"] = selects_larger_interval(df[model])
        df["tl_" + model + "_max"] = selects_larger_interval(df["tl_"+model])
        df["resc_tl_" + model + "_max"] = selects_larger_interval(df["resc_tl_" + model])
        all_models.extend(["tl_" + model, "resc_" + model, "resc_tl_" + model, "tl_resc_" + model, model+"_max", "tl_"+model+"_max","resc_tl_" + model+ "_max"])

        print("Done!")
    return all_models

def load_mesa_PSG(filename, ground_truth="stage"):
    """
        Load Groundtruth information about sleep vs active
    """
    df = pd.read_csv(filename)

    df["actValue"] = df["activity"]
    df["time"] = pd.to_datetime(df["linetime"])

    if ground_truth == "stage":
        df["gt"] = df["stage"] > 0
    elif ground_truth == "interval":
        df["gt"] = (df["interval"] != "ACTIVE").astype(int)

    df = df[df["interval"] != "EXCLUDED"] # Need to check if this is the best to do it
    df["active"] = (df["interval"] == "ACTIVE").astype(int)

    return df

def summary_table(results):
    """
        Function to make summary statistics such as mean and standard deviation for various evaluation metrics such as
        accuracy, precision, recall etc
    """
    values = {}

    values["algs"] = []
    for alg in results:
        values["algs"].append(alg)

    #values["ScoredMin"] = []
    values["SEWholeDF"] = []
    values["SEGTBlock"] = []
    values["SESelfBlock"] = []
    values["SESelfBlock5Min"] = []

    values["TotalSleep"] = []
    values["TotalSleepBlock"] = []
    values["PercentSleep"] = []
    values["PercentSleepBlock"] = []
    values["DeltaStartBlock"] = []
    values["DeltaEndBlock"] = []

    for standardMetric in ["Accuracy", "Precision", "Recall", "F1", "~F1", "Specificity"]:
        values[standardMetric] = []
        values[standardMetric + "Block"] = []

    print_format  = "%.1f +- %.1f"
    for alg in results:
        #values["ScoredMin"].append("%.0f +- %.0f" % (results[alg]["ScoredMin"].mean(), results[alg]["ScoredMin"].std()))
        values["SEWholeDF"].append(print_format % (100.*results[alg]["EfficiencyWholeDF"].mean(), 100.*results[alg]["EfficiencyWholeDF"].std()))
        values["SEGTBlock"].append(print_format % (100.*results[alg]["EfficiencyGTBlock"].mean(), 100.*results[alg]["EfficiencyGTBlock"].std()))
        values["SESelfBlock"].append(print_format % (100.*results[alg]["EfficiencySelfBlock"].mean(), 100.*results[alg]["EfficiencySelfBlock"].std()))
        values["SESelfBlock5Min"].append(print_format % (100.*results[alg]["EfficiencySelfBlock5min"].mean(), 100.*results[alg]["EfficiencySelfBlock5min"].std()))

        values["TotalSleep"].append(print_format % (results[alg]["TotalSleep"].mean(), results[alg]["TotalSleep"].std()))
        values["TotalSleepBlock"].append(print_format % (results[alg]["TotalSleepBlock"].mean(), results[alg]["TotalSleepBlock"].std()))
        values["PercentSleep"].append(print_format % (100.*results[alg]["PercentSleep"].mean(), 100.*results[alg]["PercentSleep"].std()))
        values["PercentSleepBlock"].append(print_format % (100.*results[alg]["PercentSleepBlock"].mean(), 100.*results[alg]["PercentSleepBlock"].std()))

        values["Accuracy"].append(print_format % (100.*results[alg]["AccAlg"].mean(), 100.*results[alg]["AccAlg"].std()))
        values["AccuracyBlock"].append(print_format % (100.*results[alg]["AccBlock"].mean(), 100.*results[alg]["AccBlock"].std()))
        values["Precision"].append(print_format % (100.*results[alg]["PrecAlg"].mean(), 100.*results[alg]["PrecAlg"].std()))
        values["PrecisionBlock"].append(print_format % (100.*results[alg]["PrecBlock"].mean(), 100.*results[alg]["PrecBlock"].std()))
        values["Recall"].append(print_format % (100.*results[alg]["RecAlg"].mean(), 100.*results[alg]["RecAlg"].std()))
        values["RecallBlock"].append(print_format % (100.*results[alg]["RecBlock"].mean(), 100.*results[alg]["RecBlock"].std()))
        values["F1"].append(print_format % (100.*results[alg]["F1Alg"].mean(), 100.*results[alg]["F1Alg"].std()))
        values["F1Block"].append(print_format % (100.*results[alg]["F1Block"].mean(), 100.*results[alg]["F1Block"].std()))
        values["~F1"].append(print_format % (100.*results[alg]["~F1Alg"].mean(), 100.*results[alg]["~F1Alg"].std()))
        values["~F1Block"].append(print_format % (100.*results[alg]["~F1Block"].mean(), 100.*results[alg]["~F1Block"].std()))
        values["Specificity"].append(print_format % (100.*results[alg]["SpecAlg"].mean(), 100.*results[alg]["SpecAlg"].std()))
        values["SpecificityBlock"].append(print_format % (100.*results[alg]["SpecBlock"].mean(), 100.*results[alg]["SpecBlock"].std()))

        values["DeltaStartBlock"].append("%s +- %s" % (results[alg]["DeltaStart"].mean().seconds, results[alg]["DeltaStart"].std().seconds))
        values["DeltaEndBlock"].append("%s +- %s" % (results[alg]["DeltaEnd"].mean().seconds, results[alg]["DeltaEnd"].std().seconds))

        #values["SleepEfficiency"].append("%.0f +- %.0f" % (results[alg]["SleepEfficiency"].mean(), results[alg]["SleepEfficiency"].std()))
        #values["ScoredMin_std"].append()

    return pd.DataFrame(values).set_index("algs")

def annotateSleep(d):

    d["_noActivity_p1"] = d["_noActivity"].shift(1)
    d["_sleepStarts"] = (d["_noActivity"] == True) & (d["_noActivity_p1"] == False)

    d["_cumsleep"] = d["_noActivity"].cumsum()
    d["_cumsleep_diff"] = (d["_noActivity"].cumsum().where(d["_sleepStarts"], np.nan) - 1.).fillna(method="pad").fillna(0.0)
    d["_sleepmins"] = d["_cumsleep"] - d["_cumsleep_diff"]

    d["_sleepmins"] = d["_sleepmins"].where(d["_noActivity"], 0.0 )

    del d["_noActivity_p1"]
    del d["_cumsleep"]
    del d["_cumsleep_diff"]
    del d["_sleepStarts"]

def annotateAwake(d):
    d["_activity"] = ~d["_noActivity"]

    d["_activity_p1"] = d["_activity"].shift(1)
    d["_awakeStarts"] = (d["_activity"] == True) & (d["_activity_p1"] == False)

    d["_cumawake"] = d["_activity"].cumsum()
    d["_cumawake_diff"] = (d["_activity"].cumsum().where(d["_awakeStarts"], np.nan) - 1.).fillna(method="pad").fillna(0.0)
    d["_awakemins"] = d["_cumawake"] - d["_cumawake_diff"]

    d["_awakemins"] = d["_awakemins"].where(d["_activity"], 0.0 )

    del d["_activity_p1"]
    del d["_cumawake"]
    del d["_cumawake_diff"]
    del d["_awakeStarts"]

def define_state(df):
    state = np.nan

    if (df["_sleep+"] == 0) and (df["_awaken+"] == 0):
        state = np.nan
    elif (df["_sleep+"] == 1) and (df["_awaken+"] == 0):
        state = "_sleeping"
    elif (df["_sleep+"] == 0) and (df["_awaken+"] == 1):
        state = "_awaken"
    elif (df["_sleep+"] == 1) and (df["_awaken+"] == 1):
        state = "_error"

    return state

def set_sleep_thresholds(df, min_sleep, min_awaken):
    df["_sleep+"] = (df["_sleepmins"] >= min_sleep).astype(int)
    df["_awaken+"] = (df["_awakemins"] >= min_awaken).astype(int)

    result = df[["_sleep+", "_awaken+"]].apply(define_state, axis=1).fillna(method="pad").fillna("_awaken")
    result = result.replace("_sleeping", 1).replace("_awaken",0).replace("_error", -100)

    del df["_sleep+"]
    del df["_awaken+"]

    return result.astype(np.int)

def time_based(df,  min_sleep=15, min_awaken=30):
    """
        Function used to different sleep from active using a pre-defined number of
        minutes to sleep and to wake.
    """

    df["_noActivity"] = df["actValue"] == 0
    annotateAwake(df)
    annotateSleep(df)

    result = set_sleep_thresholds(df, min_sleep, min_awaken)

    del df[u'_awakemins']
    del df[u'_sleepmins']
    del df[u'_activity']
    del df["_noActivity"]

    return result

def sazonov2(df):
    """
        Sazonov formula as shown in Tilmanne et al. 2009 paper
    """
    for w in range(1,10):
        df["_w%d" % (w-1)] = df["actValue"].rolling(window=w, min_periods=1).max()

    sazonov = 1.99604  - 0.1945 * df["_w0"] - 0.09746 * df["_w1"] - 0.09975 * df["_w2"] - 0.10194 * df["_w3"] - 0.08917 * df["_w4"] - 0.08108 * df["_w5"] - 0.07494 * df["_w6"] - 0.07300 * df["_w7"] - 0.10207 * df["_w8"]

    for w in range(1,10):
        del df["_w%d" % (w-1)]

    sazonov = 1 / (1 + np.exp(-sazonov))

    #return (sazonov >= 0.5).astype(int)
    return sazonov, (sazonov >= 0.5).astype(int)

def kripke(df, scaler = 0.204):
    """
        Kripke formula as shown in Kripke et al. 2010 paper
    """
    for i in range(1,11):
        df["_a-%d" % (i)] = df["activity"].shift(i).fillna(0.0)
        df["_a+%d" % (i)] = df["activity"].shift(-i).fillna(0.0)

    kripke = scaler * (0.0064 * df["_a-10"] + 0.0074 * df["_a-9"] + 0.0112 * df["_a-8"] + 0.0112 * df["_a-7"] + 0.0118 * df["_a-6"] + 0.0118 * df["_a-5"] + 0.0128 * df["_a-4"] + 0.0188 * df["_a-3"] + 0.0280 * df["_a-2"] + 0.0664 * df["_a-1"] + 0.0300 * df["activity"] + 0.0112 * df["_a+1"] + 0.0100 * df["_a+2"])

    for i in range(1,11):
        del df["_a+%d" % (i)]
        del df["_a-%d" % (i)]

    #return (kripke < 1.0).astype(int)
    return kripke, (kripke < 1.0).astype(int)

def sazonov(df):
    """
        Sazonov formula as shown in the original paper
    """
    for w in range(1,6):
        df["_w%d" % (w-1)] = df["actValue"].rolling(window=w, min_periods=1).max()

    sazonov = 1.727  - 0.256 * df["_w0"] - 0.154 * df["_w1"] - 0.136 * df["_w2"] - 0.140 * df["_w3"] - 0.176 * df["_w4"]

    for w in range(1,6):
        del df["_w%d" % (w-1)]

    #return (sazonov >= 0.5).astype(int)
    return sazonov, (sazonov >= 0.5).astype(int)


def sadeh(df, min_value=0):
    """
        Sadeh model for classifying sleep vs active
    """
    window_past = 6
    window_nat = 11
    window_centered = 11

    df["_mean"] = df["actValue"].rolling(window=window_centered, center=True, min_periods=1).mean()
    df["_std"] = df["actValue"].rolling(window=window_past, min_periods=1).std()
    df["_nat"] = ((df["actValue"] >= 50) & (df["actValue"] < 100)).rolling(window=window_nat, center=True, min_periods=1).sum()

    df["_LocAct"] = (df["actValue"] + 1.).apply(np.log)

    sadeh = (7.601 - 0.065 * df["_mean"] - 0.056 * df["_std"] - 0.0703 * df["_LocAct"] - 1.08 * df["_nat"])

    del df["_mean"]
    del df["_std"]
    del df["_nat"]
    del df["_LocAct"]

    #return (sadeh > min_value).astype(int)
    return sadeh, (sadeh > min_value).astype(int)

def oakley(df, threshold=80):
    """
        Oakley method to class sleep vs active/awake
    """
    for i in range(1,5):
        df["_a-%d" % (i)] = df["activity"].shift(i).fillna(0.0)
        df["_a+%d" % (i)] = df["activity"].shift(-i).fillna(0.0)

    oakley = 0.04 * df["_a-4"] + 0.04 * df["_a-3"] + 0.20 * df["_a-2"] + 0.20 * df["_a-1"] + \
             2.0 * df["activity"] + \
             0.20 * df["_a+1"] + 0.20 * df["_a-2"] + 0.04 * df["_a-3"] + 0.04 * df["_a-4"]

    for i in range(1,5):
        del df["_a+%d" % (i)]
        del df["_a-%d" % (i)]

    #return (oakley <= threshold).astype(int)
    return oakley, (oakley <= threshold).astype(int)

def tudor_locke(s, min_minutes_sleep = 5, min_minutes_awake = 10):
    """
    ****
    TODO: missing time in between sleep onset and awake onset. In the original paper it is 160 minutes.
    ****

    Tudor-Locke algorithm is based on the definition that multiple 'awake' and 'sleeping' periods
    are allowed in a sleeping epoch. It aims to define pontual bedtime and wake_time based on simple rules.

    The default implementation uses:
       bedtime = After 5 minutes of no moviment
       awaketime = after 10 minutes of moviment
    """

    bedtime = s.rolling(window=min_minutes_sleep, center=False, min_periods=1).sum() == min_minutes_sleep
    awaketime = (~s.astype(bool)).rolling(window=min_minutes_awake, center=False, min_periods=1).sum() == min_minutes_awake

    bedtime = bedtime.replace(False, np.nan) + 1
    awaketime = awaketime.replace(False, np.nan)

    returncol = bedtime.combine(awaketime, lambda x1, x2: x1 if not np.isnan(x1) else x2)
    returncol = returncol.fillna(method="ffill")

    returncol = returncol - 1
    returncol.fillna(0, inplace=True)
    return returncol.astype(int)

def webster(df):
    """
        Webster method to classify sleep from awake
    """
    df["_A0"] = df["actValue"]
    for i in range(1,5):
        df["_A-%d" % (i)] = df["actValue"].shift(i).fillna(0.0)
    for i in range(1,3):
        df["_A+%d" % (i)] = df["actValue"].shift(-i).fillna(0.0)

    w_m4, w_m3, w_m2, w_m1, w_0, w_p1, w_p2 = [0.15, 0.15, 0.15, 0.08, 0.21, 0.12, 0.13]
    p = 0.025

    webster = p * (w_m4 * df["_A-4"] + w_m3 * df["_A-3"] + w_m2 * df["_A-2"] + w_m1 * df["_A-1"] + w_0 * df["_A0"] + w_p1 * df["_A+1"] + w_p2 * df["_A+2"])

    # Remove temporary variables
    del df["_A0"]
    for i in range(1,5):
        del df["_A-%d" % (i)]
    for i in range(1,3):
        del df["_A+%d" % (i)]

    #return (webster < 1.0).astype(int)
    return webster, (webster < 1.0).astype(int)

def cole(df):
    """
        Cole method to classify sleep vs awake
    """
    df["_A0"] = df["actValue"]
    for i in range(1,5):
        df["_A-%d" % (i)] = df["actValue"].shift(i).fillna(0.0)
    for i in range(1,3):
        df["_A+%d" % (i)] = df["actValue"].shift(-i).fillna(0.0)

    w_m4, w_m3, w_m2, w_m1, w_0, w_p1, w_p2 = [404, 598, 326, 441, 1408, 508, 350]
    p = 0.00001

    cole = p * (w_m4 * df["_A-4"] + w_m3 * df["_A-3"] + w_m2 * df["_A-2"] + w_m1 * df["_A-1"] + w_0 * df["_A0"] + w_p1 * df["_A+1"] + w_p2 * df["_A+2"])

    # Remove temporary variables
    del df["_A0"]
    for i in range(1,5):
        del df["_A-%d" % (i)]
    for i in range(1,3):
        del df["_A+%d" % (i)]

    #return (cole < 1.0).astype(int)
    return cole, (cole < 1.0).astype(int)

def non_wear_choi11(df):
    # TODO: still needs testing and validation
    df["_activity60win"] = df["_activity"].rolling(window=61, center=True, min_periods=1).sum()
    df["_notWearingMin"] = df["_activity60win"] <= 2

    df["_notWearing"] = df["_notWearingMin"].rolling(window=90, center=False, min_periods=1).sum()

    df["notWearingDevice"] = (df["_notWearing"] == 90).astype(int)

    del df["_notWearing"]
    del df["_activity60win"]
    del df["_notWearingMin"]

def min_run_length(series):
    terminal = pd.Series([0])
    diffs = pd.concat([terminal, series, terminal]).diff()
    starts = np.where(diffs == 1)
    ends = np.where(diffs == -1)
    return [(e-s, (s, e-1)) for s, e in zip(starts[0], ends[0])
            if e - s >= 2]

def selects_larger_interval(s):
    intervals = min_run_length(s)
    intervals = sorted(intervals, key= lambda x : x[0], reverse=True)
    #print intervals

    if not intervals:
        # Could not find any interval. Just return
        return pd.Series(data=0, index=s.index)

    start, end = intervals[0][1]

    result = pd.Series(data=0, index=s.index)
    result.loc[start:end] = 1

    return result

def webster_rescoring_rules(s, rescoring_rules="abcde"):

    haveAppliedAnyOtherRule = False

    if "a" in rescoring_rules or "A" in rescoring_rules:
        # After at least 4 minutes scored as wake, next minute scored as sleep is rescored wake
        #print "Processing rule A"
        maskA = s.shift(1).rolling(window=4, center=False, min_periods=1).sum() > 0 # avoid including actual period
        result = s.where(maskA, 0)
        haveAppliedAnyOtherRule = True

    if "b" in rescoring_rules or "B" in rescoring_rules:
        # After at least 10 minutes scored as wake, the next 3 minutes scored as sleep are rescored wake
        #print "Processing rule B"
        if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
            s = result

        maskB = s.shift(1).rolling(window=10, center=False, min_periods=1).sum() > 0 # avoid including actual period
        result = s.where(maskB, 0).where(maskB.shift(1), 0).where(maskB.shift(2), 0)
        haveAppliedAnyOtherRule = True

    if "c" in rescoring_rules or "C" in rescoring_rules:
        # After at least 15 minutes scored as wake, the next 4 minutes scored as sleep are rescored as wake
        #print "Processing rule C"
        if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
            s = result

        maskC = s.shift(1).rolling(window=15, center=False, min_periods=1).sum() > 0 # avoid including actual period
        result = s.where(maskC, 0).where(maskC.shift(1), 0).where(maskC.shift(2), 0).where(maskC.shift(3), 0)
        haveAppliedAnyOtherRule = True

    if "d" in rescoring_rules or "D" in rescoring_rules:
        # 6 minutes or less scored as sleep surroundeed by at least 10 minutes (before or after) scored as wake are rescored wake
        #print "Processing rule D"
        if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
            s = result

        # First Part
        maskD1 = s.shift(1).rolling(window=10, center=False, min_periods=1).sum() > 0 # avoid including actual period
        tmpD1 = s.where(maskD1.shift(5), 0)
        haveAppliedAnyOtherRule = True

        # Second Part: sum the next 10 periods and replaces previous 6 in case they are all 0's
        maskD2 = s.shift(-10).rolling(window=10, center=False, min_periods=1).sum() > 0 # avoid including actual period
        tmpD2 = s.where(maskD2.shift(-5), 0)

        result = tmpD1 & tmpD2

    if "e" in rescoring_rules or "E" in rescoring_rules:
        # 10 minutes or less scored as sleep surrounded by at least 20 minutes (before or after) scored as wake are rescored wake
        #print "Processing rule E"
        if haveAppliedAnyOtherRule == True: # if this is true, I need to apply the next operation on the destination col
            s = result

        # First Part
        maskE1 = s.shift(1).rolling(window=20, center=False, min_periods=1).sum() > 0 # avoid including actual period
        tmpE1 = s.where(maskE1.shift(9), 0)

        # Second Part: sum the next 10 periods and replaces previous 6 in case they are all 0's
        maskE2 = s.shift(-20).rolling(window=20, center=False, min_periods=1).sum() > 0 # avoid including actual period
        tmpE2 = s.where(maskE2.shift(-9), 0)

        result = tmpE1 & tmpE2

    return result

def onset_after_X_minutes(s, X):
    """
      Cole 92 defines this period as:
      "sleep onset is the beginning of the first interval containing at least n minutes
      scored as sleep stage 1 or greated with no more than 1 minute of wakefulness intervening
    """
    __onset_candidate = s.rolling(window=X, center=False, min_periods=1).sum() >= (X-1)

    # Gets the index of the first candidate...
    if __onset_candidate.empty:
        result = pd.Series(data=0, index=s.index)
        return s.shape[0] - 1

    # If there is no single candidate, returns.
    else:
        idx = __onset_candidate.idxmax() # Returns number to be used with 'df.loc' function

    result = pd.Series(data=0, index=s.index)
    #print "ONSET 1:", s.shape[0]

    start_ilocation = s.index.get_loc(idx)
    result.iloc[start_ilocation - (X - 2)] = 1
    #print "ONSET: idx: %d, idx - X: %d " % (idx, idx-(X-2))
    #print "ONSET 2:", s.shape[0]
    return start_ilocation - (X - 2) # Returns the .iloc of the object

def twu_after_X_minutes(s, onset_idx, X):
    """
    TWU stands for Time woke up (terrible name created by me)

    Inspired in the Onset time definition made by Cole 92:
      "sleep onset is the beginning of the first interval containing at least n minutes
      scored as sleep stage 1 or greated with no more than 1 minute of wakefulness intervening
    """

    # Transforms everything before onset_idx into NAN values
    __filter = ~s.astype(bool)
    __filter.loc[(s.index < s.index[onset_idx])] = np.nan

    # Find candidates
    __twu_candidate = __filter.rolling(window=X, center=False, min_periods=1).sum() >= (X-1)

    # Gets the index of the first candidate... (already filtered everything before onset_idx)
    if __twu_candidate.empty:
        result = pd.Series(data=0, index=s.index)
        #print "RETURNING the last position in TWU"
        return result.shape[0] - 1 # TODO: maybe change to result.index[-1]
    else:
        idx = __twu_candidate.idxmax()

    del __twu_candidate

    result = pd.Series(data=0, index=s.index)
    #print "IN TWU 1:", df.shape[0]

    start_ilocation = s.index.get_loc(idx)
    result.iloc[start_ilocation - (X - 2)] = 1
    #print "TWU: idx: %d, idx - X: %d " % (start_ilocation, start_ilocation - (X - 2))
    #print "IN TWU 2:", df.shape[0]
    return start_ilocation - (X - 2)

def make_one_block(s, start_idx, end_idx):
    """
    Start_idx and end_idx are LABELS (can be any data type), not position (only integers)
    """
    result = pd.Series(data=0, index=s.index)
    result.iloc[start_idx:end_idx] = 1
    return result

def make_sleep_block(s, X_onset, X_twu):
    """
    Usage: e.g. make_sleep_block(df, alg="sadeh", X_onset=20, X_twu=40, newcol="sadeh_block")

    start_idx = onset_after_X_minutes(df, alg, "onset_" + alg, X = X_onset)
    end_idx = twu_after_X_minutes(df, col=alg, onset_idx=start_idx, newcol="twu_" + alg, X=X_twu)
    df[newcol] = make_one_block(df[alg], start_idx, end_idx)
    """
    #print "Before onset:", df.shape[0]
    start_idx = onset_after_X_minutes(s, X = X_onset)

    #print "Before TWU:", df.shape[0]
    end_idx = twu_after_X_minutes(s, onset_idx=start_idx, X=X_twu)

    #print "Before finishing make_sleep_block. START: %d, END: %d, Shape: %d" % (start_idx, end_idx, df.shape[0])
    return make_one_block(s, start_idx, end_idx)

def print_signals(df, cols, figsize=(16,12)):
    if "gt" in cols:
        df["gt"] = df["gt"].astype(int)
    df[cols + ["time"]].plot(subplots=True, figsize=figsize, x ="time")

def get_features(df, winsize=20):

    featnames = []

    for winsize in range(1, winsize):
        df["_mean_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=False, min_periods=1).mean().fillna(0.0)
        df["_mean_centered_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=True, min_periods=1).mean().fillna(0.0)

        df["_median_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=False, min_periods=1).median().fillna(0.0)
        df["_median_centered_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=True, min_periods=1).median().fillna(0.0)

        df["_std_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=False, min_periods=1).std().fillna(0.0)
        df["_std_centered_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=True, min_periods=1).std().fillna(0.0)

        df["_max_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=False, min_periods=1).max().fillna(0.0)
        df["_max_centered_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=True, min_periods=1).max().fillna(0.0)

        df["_min_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=False, min_periods=1).min().fillna(0.0)
        df["_min_centered_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=True, min_periods=1).min().fillna(0.0)

        df["_var_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=False, min_periods=1).var().fillna(0.0)
        df["_var_centered_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=True, min_periods=1).var().fillna(0.0)

        df["_nat_%d" % (winsize)] = ((df["actValue"] >= 50) & (df["actValue"] < 100)).rolling(window=winsize, center=False, min_periods=1).sum().fillna(0.0)
        df["_nat_centered_%d" % (winsize)] = ((df["actValue"] >= 50) & (df["actValue"] < 100)).rolling(window=winsize, center=True, min_periods=1).sum().fillna(0.0)

        df["_anyact_%d" % (winsize)] = (df["actValue"] > 0).rolling(window=winsize, center=False, min_periods=1).sum().fillna(0.0)
        df["_anyact_centered_%d" % (winsize)] = (df["actValue"] > 0).rolling(window=winsize, center=True, min_periods=1).sum().fillna(0.0)

        if winsize > 3:
            df["_skew_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=False, min_periods=1).skew().fillna(0.0)
            df["_skew_centered_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=True, min_periods=1).skew().fillna(0.0)
            #
            df["_kurt_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=False, min_periods=1).kurt().fillna(0.0)
            df["_kurt_centered_%d" % (winsize)] = df["actValue"].rolling(window=winsize, center=True, min_periods=1).kurt().fillna(0.0)

        for variant in ["centered_", ""]:
            featnames.append("_mean_%s%d" % (variant, winsize))
            featnames.append("_median_%s%d" % (variant, winsize))
            featnames.append("_max_%s%d" % (variant, winsize))
            featnames.append("_min_%s%d" % (variant, winsize))
            featnames.append("_std_%s%d" % (variant, winsize))
            featnames.append("_var_%s%d" % (variant, winsize))
            featnames.append("_nat_%s%d" % (variant,winsize))
            featnames.append("_anyact_%s%d" % (variant,winsize))
            if winsize > 3:
                featnames.append("_kurt_%s%d" % (variant, winsize))
                featnames.append("_skew_%s%d" % (variant, winsize))

    df["_Act"] = (df["actValue"]).fillna(0.0)
    df["_LocAct"] = (df["actValue"] + 1.).apply(np.log).fillna(0.0)

    featnames.append("_LocAct")
    featnames.append("_Act")
    return featnames

def apply_formulas_to_psgfile(filename):
    """
        Process a PSG file with basic scoring algorithms
    """
    df = load_mesa_PSG(filename)

    df["baselinesleep"] = 1
    df["baselineawake"] = 0

    df["timebased"] = time_based(df,  min_sleep=15, min_awaken=30)

    # This is the GT block:
    gtTrue = df[df["gt"] == True]
    start_block = df.index.get_loc(gtTrue.index[0])
    end_block =  df.index.get_loc(gtTrue.index[-1])
    #print "Start:", start_block, "End:", end_block
    df["gt_sleep_block"] = make_one_block(df["gt"], start_block, end_block)

    if df[df["gt_sleep_block"] == True].empty:
        print("**** ERROR: Ops...'gt_sleep_block' should not be EMPTY")
        ERROR___
        return []

    df["p_sazonov"],df["sazonov"] = sazonov(df)
    df["p_sazonov2"],df["sazonov2"] = sazonov2(df)
    df["p_sadeh"],df["sadeh"] = sadeh(df)
    df["p_cole"],df["cole"] = cole(df)
    df["p_oakley"],df["oakley"] = oakley(df, 10)
    df["p_kripke"],df["kripke"] = kripke(df)
    df["p_webster"],df["webster"] = webster(df)

    return df

def grid_search(df, function, parameters, eval_function):
    """
        Use grid search for hyper-parameter optimization
    """
    keys = parameters.keys()
    meta_values = []
    for key in keys:
        meta_values.append( parameters[key] )
    print(meta_values)

    results = []
    combinations = list(product(*meta_values))

    print("Running %d combinations" % (len(combinations)))

    for combnum, p in enumerate(combinations):
        input_parameters = {}
        for i, _ in enumerate(keys):
            input_parameters[keys[i]] = p[i]

        print("%d - Running grid search with %s" % (combnum, input_parameters))

        grps = df.groupby("mesaid")
        #df.groupby("mesaid")["gt","actValue"].apply(lambda s: function(s, **input_parameters))
        r = []
        for grp in grps:
            tmp = grp[1].copy()
            r.append(function(tmp, **input_parameters))
            #print "Grp", grp[0], "Shape", tmp.shape

        df["grid"] = pd.concat(r)
        #df["grid"] = pd.concat(r).reset_index(drop=True).values

        result = df.groupby("mesaid")[["grid","gt"]].apply(lambda x: eval_function(x["gt"],x["grid"])).mean()
        input_parameters["result"] = result

        print("...result: %.3f" % (result))

        results.append(input_parameters)

    del df["grid"]
    return results

def resave_dftest(task):
    print("...Loading Task %d dataset into memory..." % (task))
    _, dftest, _ = load_dataset("hdf_task%d"  % (task), useCache=True)

    if "interval" in dftest and "binterval" not in dftest:
        dftest["binterval"] = dftest["interval"].replace("ACTIVE", 0).replace("REST",1).replace("REST-S", 1)

    dfoutname = "dftest_task%d.csv" % (task)
    print("...Saving Task %d dataset to disk. Filename: %s ..." % (task, dfoutname))
    dftest[["mesaid", "linetime", "marker", "interval", "binterval", "gt", "gt_sleep_block", "wake"]].to_csv(dfoutname, index=False)
    print("...Done...")

def sleeping_in_previous_X_epochs_from_idx(gt, idx, X=30):
    #print "Sum:", sum(gt.loc[idx-X:idx] > 0)
    return sum(gt.loc[idx-X:idx] > 0) < 5

def sleeping_in_next_X_epochs_from_idx(gt, idx, X=30):
    #print "Sum:", sum(gt.loc[idx:idx+X] > 0)
    return sum(gt.loc[idx:idx+X] > 0) < 5

def get_marker_positions(m, gt):
    mid = m.shape[0]/2

    #print("Possible first half:\n", m[0:mid][m[0:mid] > 0])
    candidates =  m[0:mid][m[0:mid] > 0]
    if candidates.empty:
        # Just take the first value in the interval as the marker
        idx_tail1_first_half = m.head(1).index[0]
    else:
        for i in range(1, candidates.shape[0] + 1):
            # print i
            idx_tail1_first_half = candidates.tail(i).head(1).index[0]
            if sleeping_in_previous_X_epochs_from_idx(gt, idx_tail1_first_half):
                break
    #print("Picked", idx_tail1_first_half)

    #print("Possible second half:\n", m[mid:][m[mid:] > 0])
    candidates = m[mid:][m[mid:] > 0]

    candidates = m[mid:][m[mid:] > 0]
    if candidates.empty:
        idx_head1_sec_half = m.tail(1).index[0]
    else:
        for i in range(1, candidates.shape[0] + 1):
            # print i
            idx_head1_sec_half = candidates.head(i).tail(1).index[0]
            if sleeping_in_next_X_epochs_from_idx(gt, idx_head1_sec_half):
                break

    #print("Picked", idx_head1_sec_half)

    #print("Final size", s.loc[idx_tail1_first_half:idx_head1_sec_half].shape[0], "instead of", s.shape[0])
    return idx_tail1_first_half, idx_head1_sec_half

