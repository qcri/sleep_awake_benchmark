import pandas as pd
import numpy as np
import sys

from glob import glob
import pickle
from sleep_eval import evaluate_scoring_algorithm, evaluation_summary
from sleep_misc import rescore_models
#from scipy.stats import ttest_ind

baselines = ["gt", "always1", "always0", "binterval", "wake", "sleep"]
defaultalgs = ["sazonov", "sazonov2", "cole", "time_based", "sadeh", "oakley", "kripke", "webster"]
defaultml = ["ExtraTrees", "SGD_perceptron", "SGD_log", "SGD_hinge", "SGD_huber"]
defaultnn = ["LSTM_20_raw", "LSTM_50_raw", "LSTM_100_raw", "CNN_20_raw", "CNN_50_raw", "CNN_100_raw"]
algs = defaultalgs + defaultml + defaultnn


def get_nndf(task, nn_type, feature_type):
    """
        Get the dataframe corresponding to different configurations of LSTM or CNN (Deep Learning models)
    """

    files = glob("./results/task%d_%s_%s*.csv.gz" % (task, nn_type, feature_type))

    result = []
    for file in files:
        df = pd.read_csv(file)

        nn_keys = []
        for k in df.keys():
            if nn_type in k:
                nn_keys.append(k)

        for k in nn_keys:
            df[k + "_" + feature_type] = df[k]
            del df[k]

        result.append(df)

    if len(result) == 1:
        return result[0]
    else:
        merged = pd.merge(result[0], result[1])
        for i in range(2, len(result)):
            merged = pd.merge(merged, result[i])
        return merged


def get_nns(task):
    """
    """
    lstm_raw = get_nndf(task, "LSTM", "raw")
    cnn_raw = get_nndf(task, "CNN", "raw")

    merged = pd.merge(lstm_raw, cnn_raw)
    return merged

def load_results(task):
    """
        Load results from formula, machine learning based methods and combine with  deep learning model based results
    """

    ALGRESULTS = "./results/task%d_formulas.csv.gz" % (task)
    MLRESULTS = "./results/task%d_ml.csv.gz" % (task)

    dftest = pd.read_csv("./dftest_task%d.csv" % (task))

    dfalg = pd.read_csv(ALGRESULTS)
    dfml  = pd.read_csv(MLRESULTS)
    dfnn = get_nns(task)
    dfml = dfml.rename(columns={"Unnamed: 0":"algs"})

    merged = pd.merge(dfalg, dfml, on=["mesaid","linetime","actValue","gt","gt_sleep_block"]) #
    merged = pd.merge(merged, dfnn, on=["mesaid","linetime","actValue","gt","gt_sleep_block"]) #
    merged = pd.merge(merged, dftest, on=["mesaid","linetime","gt","gt_sleep_block"]) #

    merged["time"] = pd.to_datetime(merged["linetime"])
    merged["always1"] = 1
    merged["always0"] = 0

    merged["sleep"] = (~merged["wake"].astype(np.bool)).astype(float)
    return merged


if __name__ == "__main__":
    """
        Get a summary dataframe with all the evaluation results for different predictive models
    """

    TASK = int(sys.argv[1])

    PICKLE_RESULTFILE = "task%d_results.pkl" % (TASK)
    SUMMARY_RESULTFILE = "task%d_summary.csv" % (TASK)

    df = load_results(TASK)

    print("Expanding algorithms...")
    expanded_algs = rescore_models(df, algs)

    results = {}
    for alg in baselines + expanded_algs:
        results[alg] = evaluate_scoring_algorithm(df, alg)

    #print("Example of how to get p-values:")
    #print("P = %.5f" % (ttest_ind(results1["cole"]["Precision"], results1["saznov"]["Precision"]))[1])

    summary = evaluation_summary(df, expanded_algs + baselines)
    summary = pd.DataFrame(summary)
    summary.to_csv(SUMMARY_RESULTFILE)
    print("Created summary '%s'" % (SUMMARY_RESULTFILE))

    with open(PICKLE_RESULTFILE, "w") as f:
        pickle.dump(results, f)
    print("Created pickle result file '%s'" % (PICKLE_RESULTFILE))

