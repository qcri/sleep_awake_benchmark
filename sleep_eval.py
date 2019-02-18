import pandas as pd
import numpy as np
from sleep_misc import make_sleep_block, get_marker_positions
from sklearn import metrics

EVAL_METRICS = ["SEInterval", "SEOnlyREST", "SEMarkers", "SEWholeDF", "SEGTBlock", "SESelfBlock", "SESelfBlock5Min",
                "TotalSleep", "TotalSleepBlock", "PercentSleep", "PercentSleepBlock", "DeltaStartBlock", "DeltaEndBlock",
                "Accuracy", "AccuracyBlock", "Precision", "PrecisionBlock", "Recall", "RecallBlock",
                "F1", "F1Block", "~F1", "~F1Block", "Specificity","SpecificityBlock"]
TIME_METRICS = ["DeltaStartBlock", "DeltaEndBlock"]

def evaluation_summary(df, scoring_algorithms, precomputed_dict=None):
    """
        Create a summary table with all evaluation metrics presenting their mean +/- standard deviation
    """
    mlresults = []

    for alg in scoring_algorithms:
        if precomputed_dict is not None:
            res = precomputed_dict[alg]
        else:
            res = evaluate_scoring_algorithm(df, alg)


        stds = res.std(axis=0).rename("Std")
        means = res.mean(axis=0).rename("Mean")
        res = pd.concat([means, stds], axis=1)

        a = res.loc[set(EVAL_METRICS) - set(TIME_METRICS)].apply(lambda x:"%.1f  +- %.1f " % (100.*x["Mean"], 100.*x["Std"]), axis=1).rename(alg)
        b = res.loc[set(TIME_METRICS)].apply(lambda x: "%s +- %s" % (x["Mean"].seconds, x["Std"].seconds), axis=1).rename(alg)

        mlresults.append(pd.concat((a,b)))

    return mlresults

def evaluate_scoring_algorithm(df, alg):

    df["time"] = pd.to_datetime(df["linetime"])

    print("Evaluating model %s..." % (alg))

    df[alg + "_block"] = df.groupby("mesaid")[alg].apply(lambda s: make_sleep_block(s, X_onset=20, X_twu=100))

    r = []

    if alg == "gt":
        # SEFirstAndLastREST
        r.append(df.groupby("mesaid")[["gt","interval"]].apply(lambda x: sleep_efficiency_first_last_REST(x["gt"], x["interval"])))
        # SEFOnlyREST
        r.append(df.groupby("mesaid")[["gt","interval"]].apply(lambda x: sleep_efficiency_only_with_REST(x["gt"], x["interval"])))
        # SEBetweenMarker
        r.append(df.groupby("mesaid")[["marker","gt"]].apply(lambda x: sleep_efficiency_marker(x["gt"], x["marker"], x["gt"])))
        # SEWholeDF
        r.append(df.groupby("mesaid")[["gt"]].apply(lambda x: sleep_efficiency(x["gt"], x.index[0], x.index[-1])))
    else:
        # SEFirstAndLastREST
        r.append(df.groupby("mesaid")[[alg, "interval"]].apply(lambda x: sleep_efficiency_first_last_REST(x[alg], x["interval"])))
        # SEOnlyREST
        r.append(df.groupby("mesaid")[[alg, "interval"]].apply(lambda x: sleep_efficiency_only_with_REST(x[alg], x["interval"])))
        # SEBetweenMarker
        r.append(df.groupby("mesaid")[[alg,"marker","gt"]].apply(lambda x: sleep_efficiency_marker(x[alg], x["marker"], x["gt"])))
        # SEWholeDF
        r.append(df.groupby("mesaid")[[alg,"gt"]].apply(lambda x: sleep_efficiency(x[alg], x.index[0], x.index[-1])))

    # SEGTBlock
    r.append(df.groupby("mesaid")[[alg,"gt_sleep_block"]].apply(lambda x: 0.0 if x["gt_sleep_block"].empty else sleep_efficiency(x[alg], x[x["gt_sleep_block"] > 0].index[0], x[x["gt_sleep_block"] > 0].index[-1])))

    # SESelfBlock
    r.append(df.groupby("mesaid")[[alg,alg+"_block"]].apply(lambda x: 0.0 if x[x[alg + "_block"]>0].empty else sleep_efficiency(x[alg], x[x[alg + "_block"] > 0].index[0], x[x[alg + "_block"] > 0].index[-1])))
    # SESelfBlock5Min
    r.append(df.groupby("mesaid")[["actValue", alg+"_block"]].apply(lambda x: 0.0 if x[x[alg + "_block"]>0].empty else sleep_efficiency_wo_act_more_X_min(x["actValue"], x[x[alg + "_block"] > 0].index[0], x[x[alg + "_block"] > 0].index[-1], X_epo=10)))

    # TotalSleep and TotalSleepBlock
    r.append(df.groupby("mesaid")[[alg]].apply(lambda x: total_sleep_time(x, alg)))
    r.append(df.groupby("mesaid")[[alg+"_block"]].apply(lambda x: total_sleep_time(x, alg + "_block")))

    # PercentSleep and PercentSleepBlock
    r.append(df.groupby("mesaid")[[alg]].apply(lambda x: percent_sleep(x, alg)))
    r.append(df.groupby("mesaid")[[alg+"_block"]].apply(lambda x: percent_sleep(x, alg + "_block")))

    deltas = df.groupby("mesaid")[["time","gt_sleep_block",alg+"_block"]].apply(lambda x: delta_time_block(x["time"], x["gt_sleep_block"], x[alg+"_block"]))
    r.append(deltas.apply(lambda x: x[0]))
    r.append(deltas.apply(lambda x: x[1]))

    for func in [eval_acc, eval_precision, eval_recall, eval_f1, eval_f1_awake, eval_specificity]:
        #print "Evaluating %s" % func.func_name
        if alg != "gt":
            r.append(df.groupby("mesaid")[[alg,"gt"]].apply(lambda x: func(x["gt"],x[alg])))
        else:
            v = df.groupby("mesaid")[["gt"]].apply(lambda x: func(x["gt"],x["gt"]))
            r.append(v)
        r.append(df.groupby("mesaid")[[alg + "_block","gt_sleep_block"]].apply(lambda x: func(x["gt_sleep_block"], x[alg + "_block"])))

    res = pd.concat(r, axis=1)
    res.columns = EVAL_METRICS

    return res

def minutes_scored(df):
    return df.shape[0]

def total_sleep_time(df, col):
    return df[col].sum()

def percent_sleep(df, col):
    return 1. * df[col].sum() / df.shape[0]


"""
    Definition of various evaluation metrics
"""
def eval_precision(gt, pred):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.precision_score(gt.fillna(0.0).astype(bool), pred.fillna(0.0).astype(bool), average='binary')
    else:
        return metrics.precision_score(gt, pred, average='binary')

def eval_acc(gt, pred):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.accuracy_score(gt.fillna(0.0).astype(bool), pred.fillna(0.0).astype(bool))
    else:
        return metrics.accuracy_score(gt, pred)

def eval_recall(gt, pred):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.recall_score(gt.fillna(0.0).astype(bool), pred.fillna(0.0).astype(bool))
    else:
        return metrics.recall_score(gt, pred)

def eval_specificity(gt, pred):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.recall_score(gt.fillna(0.0).astype(bool) == False, pred.fillna(0.0).astype(bool) == False)
    else:
        return metrics.recall_score(gt == False, pred == False)

def eval_f1(gt, pred):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.f1_score(gt.fillna(0.0).astype(bool), pred.fillna(0.0).astype(bool), average='binary')
    else:
        return metrics.f1_score(gt, pred, average='binary')

def eval_f1_awake(gt, pred):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.f1_score(gt.fillna(0.0).astype(bool) == False, pred.fillna(0.0).astype(bool) == False, average='binary')
    else:
        return metrics.f1_score(gt == False, pred == False, average='binary')


def sleep_efficiency(s, start, end):
    """
    Start and end are the location index (locations are given by .index[]).
    """
    #print "Start: %d, End: %d" % (start,end)
    if end-start > 0:
        return 1. * s.loc[start:end].sum() / (end-start)
    else:
        return np.nan


def sleep_efficiency_first_last_REST(s, interval):
    resting = interval[interval != "ACTIVE"]

    if resting.empty:
        print("ERROR: could not find a REST period in the interval")
        return np.nan

    first = resting.head(1).index[0]
    last = resting.tail(1).index[0]
    return sleep_efficiency(s, first, last)

def sleep_efficiency_only_with_REST(s, interval):
    resting = interval[interval != "ACTIVE"]

    if resting.empty:
       print("ERROR: could not find a REST period in the interval")
       return np.nan

    focus_period = s.loc[resting.index]
    return 1.0 * focus_period.sum() / focus_period.shape[0]


def sleep_efficiency_marker(s, m, gt):
    """
    Input parameters:
        * s : a vector with predictions for each one of the epoches (1: sleep, 0: wake)
        * m : a vector representing the marker (1: pressed, 0: not pressed)

    This function calculates the sleep efficient in between two periods:
        (1) the last time the marker was pressed in the first half of the period
        (2) the first time the maker in the second half of the period
    """
    start, end = get_marker_positions(m, gt)
    return sleep_efficiency(s, start, end)


def sleep_efficiency_wo_act_more_X_min(actCol, start, end, X_epo):
    """
    Calculates sleep efficiency as the number of slept minutes during [start:end] interval.
    Removes from the amount of slept minutes all periods that are greater than X_epo of activity
    """
    sslice = actCol.loc[start:end].copy()

    __moreThanX = (sslice > 0).rolling(window=X_epo, center=True, min_periods=0).sum() >= X_epo
    __ignorePeriod = __moreThanX.rolling(window=X_epo, center=True, min_periods=1).sum() >= 1.0

    return 1.0 - (1.* __ignorePeriod.sum() / __ignorePeriod.shape[0])

def delta_time_block(time, gt, algdf):

    gtTrue = gt[gt == True]
    if gtTrue.empty:
        print("Ops...ground truth should not be empty")
    start_gt, end_gt = gtTrue.index[0], gtTrue.index[-1]

    algTrue = algdf[algdf == True]
    if algTrue.empty:
        # print("Ops...alg block is empty")
        start_alg, end_alg = algdf.index[0], algdf.index[-1]
    else:
        start_alg, end_alg = algTrue.index[0], algTrue.index[-1]

    return np.abs(time.loc[start_gt] - time.loc[start_alg]), np.abs(time.loc[end_gt] - time.loc[end_alg])


