import pandas as pd
import numpy as np
import pickle

from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sleep_misc import load_dataset

TASKID = 1
INPUTFILE = "hdf_task%d" % (TASKID)
EXPERIMENT = "gbc"
SCORING = "accuracy" # f1, accuracy, precision
OUTPUT = "task%d_%s_%s" % (TASKID, EXPERIMENT, SCORING)

print("...Loading dataset into memory...")
dftrain, dftest, featnames = load_dataset(INPUTFILE, useCache=True)
print("...Done...")

classifier = None

if EXPERIMENT == "sgd":
    classifier = SGDClassifier(random_state=42, n_jobs=1)
elif EXPERIMENT == "etc":
    classifier = ExtraTreesClassifier(random_state=42, n_jobs=8)
elif EXPERIMENT == "gbc":
    classifier = GradientBoostingClassifier(random_state=42)

etc_params = [
  {'classifier__n_estimators': [8, 64, 128, 512, 1024],
   'classifier__criterion': ['gini', 'entropy'],
   'classifier__class_weight': [None, "balanced", "balanced_subsample"],
   'classifier__max_depth': [None, 10, 20],
   'classifier__min_impurity_decrease': [0.0, 0.1, 0.2],
  }
]

# Currently 16200 fits
gbc_params = [
  {'classifier__loss': ['deviance', 'exponential'],
   'classifier__criterion': ['friedman_mse', 'mse', 'mae'],
   'classifier__subsample': [1.0, 0.8],
   'classifier__n_estimators': [8, 64, 128, 512, 1024],
   'classifier__max_depth': [3, 5, 10],
   'classifier__max_leaf_nodes': [None, 3, 10],
   'classifier__learning_rate': [0.1, 0.3, 0.5],
   'classifier__warm_start': [False, True],
  }
]

sgd_params = [
  {'classifier__loss': ['hinge', 'log', 'modified_huber', 'perceptron', 'squared_loss', 'huber'],
   'classifier__penalty': ['l1','l2','elasticnet'],
   'classifier__fit_intercept': [True, False],
   'classifier__max_iter': [5, 10, 20],
   'classifier__class_weight': ["balanced",None],
   'classifier__warm_start': [False, True],
   'classifier__alpha': 10.0**-np.arange(1,7)
  }
]

if EXPERIMENT == "sgd":
    params = sgd_params
elif EXPERIMENT == "etc":
    params = etc_params
elif EXPERIMENT == "gbc":
    params = gbc_params

pipe = Pipeline([
    ('normalizer', StandardScaler()),
    ('classifier', classifier)
])

def optimize_parameters(dftrain, featnames, pipe, parames, scoring, experiment):
    if experiment == "etc":
        gs = GridSearchCV(pipe, param_grid=params, n_jobs=5, cv=5, scoring=scoring, verbose=10, pre_dispatch=10)
    else:
        gs = GridSearchCV(pipe, param_grid=params, n_jobs=-1, cv=5, scoring=scoring, verbose=10)

    gs.fit(dftrain[featnames], dftrain["gt"])
    print(gs.best_estimator_)
    return gs

def display_paramters(gs):
    gsres = pd.DataFrame(gs.cv_results_['params'])
    gsres["train"] = gs.cv_results_['mean_train_score']
    gsres["test"] = gs.cv_results_['mean_test_score']
    return gsres.sort_values("test", ascending=False)

print("...Optmizing paramters for %s..." % (pipe.get_params()["classifier"]))
print("...Scoring function is %s..." % (SCORING))

gs = optimize_parameters(dftrain, featnames, pipe, params, SCORING, EXPERIMENT)
gsres = display_paramters(gs)

with open("gs_" + OUTPUT + ".pk", "wb") as f:
    pickle.dump(gs, f)

predictions = gs.predict(dftest[featnames])
print(" - Acuracy: %.4f" % (metrics.accuracy_score(dftest["gt"], predictions)))
print(" - F1: %.4f" % (metrics.f1_score(dftest["gt"], predictions)))
print(" - Best parameter: %s" % (gs.best_estimator_))

gsres.to_csv(OUTPUT + ".csv")
print("...Generated file '%s'..." % (OUTPUT))
print("...All done...")
