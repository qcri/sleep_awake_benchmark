import numpy as np
import sys
import pickle

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from sleep_misc import load_dataset

TASK = int(sys.argv[1])

DATASET_PATH = "hdf_task%d" % (TASK)
OUTPUT = "task%d_ml.csv" % (TASK)
SUMMARY_OUTPUT = "task%d_summary_ml.csv" % (TASK)
SAVED_MODELS_OUTPUT = "model_ml_task%d.pkl" % (TASK)

print("...Loading Task %d dataset into memory..." % (TASK))
dftrain, dftest, featnames = load_dataset(DATASET_PATH, useCache=True)
print("...Done...")

# Optmized models for Task 1:
models = [
     ("SGD_log", SGDClassifier(loss='log', alpha=0.0001, class_weight=None, fit_intercept=False, penalty='l1', warm_start=True, l1_ratio=0.15, max_iter=20, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5,   average=False, n_iter=None)),
     ("SGD_modhuber", SGDClassifier(loss='modified_huber', alpha=0.01, class_weight=None, fit_intercept=False, penalty='l2', warm_start=False, l1_ratio=0.15, max_iter=20, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5,   average=False, n_iter=None)),
     ("SGD_hinge",SGDClassifier(loss='hinge', alpha=0.001, class_weight="balanced", fit_intercept=False, penalty='elasticnet', warm_start=False, l1_ratio=0.15, max_iter=10, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5,   average=False, n_iter=None)),
     ("SGD_huber",SGDClassifier(loss='huber', alpha=0.00001, class_weight="balanced", fit_intercept=True, penalty='l1', warm_start=False, l1_ratio=0.15, max_iter=20, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5,   average=False, n_iter=None)),
     ("SGD_perceptron",SGDClassifier(loss='perceptron', alpha=0.01, class_weight="balanced", fit_intercept=False, penalty='elasticnet', warm_start=True, l1_ratio=0.15, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=42, learning_rate='optimal', eta0=0.0, power_t=0.5,   average=False, n_iter=None)),
     ("ExtraTrees", ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy', max_depth=20, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=512, n_jobs=8, oob_score=False, random_state=None, verbose=0, warm_start=False)) # Acc:0.8277, F1:0.8633
]


def run_test(model, Xtrain, Xtest, Ytrain, Ytest, model_name):
    model.fit(Xtrain,Ytrain)

    # Probabilities are not available for some models
    if model_name in ["SGD_hinge", "SGD_perceptron", "SGD_huber"]:
        prob = np.empty((Xtest.shape[0],))
    else:
        prob = model.predict_proba(Xtest)[:,1] # Saving only the probability of the 'sleep' class

    pred = model.predict(Xtest)
    print(" - Final Acuracy: %.4f" % (metrics.accuracy_score(Ytest, pred)))
    print(" - Final F1: %.4f" % (metrics.f1_score(Ytest, pred)))
    return model, prob, pred

# Run models:
print("...Preparing dataset...")

Xtrain = dftrain[featnames].values
Xtest = dftest[featnames].values
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

Ytrain = dftrain["gt"].values
Ytest = dftest["gt"].values

saved_models = []

print("...Runing models...")
for (model_name, model) in models:
    print("...Model: %s..." % (model))
    model, dftest["p_" + model_name], dftest[model_name] = run_test(model, Xtrain, Xtest, Ytrain, Ytest, model_name)
    saved_models.append([model_name, model])
    print("...Done with %s..." % (model_name))
print("...Done...")

print("...Saving task%d_ml.csv..." % (TASK))
#Creating a table with output predictions for all the different ML methods
dftest[[m[0] for m in models]] = dftest[[m[0] for m in models]].astype(float)
dftest["gt_sleep_block"] = dftest["gt_sleep_block"].astype(int)
dftest["gt"] = dftest["gt"].astype(int)
dftest["actValue"] = dftest["actValue"].fillna(0.0).astype(int)
dftest[["mesaid","linetime","actValue","gt","gt_sleep_block"] + [m[0] for m in models] + ["p_" + m[0] for m in models]].to_csv("task%d_ml.csv" % (TASK), index=False)
print("...Done...")

dict_model = {"scaler": scaler, "models": saved_models}
print("...Saving trained models for Task%d..." % (TASK))
with open(SAVED_MODELS_OUTPUT, "wb") as f:
    pickle.dump(dict_model, f)

