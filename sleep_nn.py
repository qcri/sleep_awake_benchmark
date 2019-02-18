import numpy as np
import sys
import time
#
from sleep_misc import load_dataset
from sleep_eval import eval_f1, eval_acc
from sklearn.preprocessing import StandardScaler
#
from keras.layers import Flatten, Convolution1D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.models import Sequential
from keras.models import load_model
from keras import layers
from keras.layers import Dense

from sklearn.model_selection import train_test_split

TASK = int(sys.argv[1])
SEQ_LEN = int(sys.argv[2]) # used in the experiments 20, 50 and 100
NN_TYPE = sys.argv[3] # LSTM or CNN
TRAINING = True
np.random.seed(42)

# Parameters used in the experiments
input_type = "raw"
epochs = 10
batch_size = 32

MODEL_OUTFILE = "model_%s_task%d_%s_seq%d.pkl" % (NN_TYPE, TASK, input_type, SEQ_LEN)
RESULTS_OUTFILE = "task%d_%s_%s_%d.csv" % (TASK, NN_TYPE, input_type, SEQ_LEN)

DATASET_PATH = "hdf_task%d" % (TASK)

print("...Loading dataset into memory...")
dftrain, dftest, featnames = load_dataset(DATASET_PATH, useCache=True)
print("...Done...")

scaler = StandardScaler()
scaler.fit(dftrain[["activity"]].fillna(0.0))

dftrain["activity"] = scaler.transform(dftrain[["activity"]].fillna(0.0))
dftest["activity"] = scaler.transform(dftest[["activity"]].fillna(0.0))

def extract_x_y(df, seq_len, mesaid):
    df = df[df["mesaid"] == mesaid][["activity","gt"]].copy()

    for s in range(1,seq_len/2 + 1):
	    df["shift_%d" % (s)] = df["activity"].shift(s)

    for s in range(1,seq_len/2 + 1):
	    df["shift_-%d" % (s)] = df["activity"].shift(-s)

    y = df["gt"]
    y = np.array([[1] if v else [0] for v in y])
    del df["gt"]
    x = df.fillna(-1).values
    return x,y

def get_data(df, seq_len):
    mesaids = df.mesaid.unique()
    x_, y_ = extract_x_y(df, seq_len, mesaids[0])

    for mid in mesaids[1:]:
        x_tmp, y_tmp = extract_x_y(df, seq_len, mid)
        x_ = np.concatenate((x_, x_tmp))
        y_ = np.concatenate((y_, y_tmp))
    return x_, y_

def build_model(input_dim, nn_type):
    if nn_type == "CNN":
        return build_model_CNN(input_dim)
    elif nn_type == "LSTM":
        return build_model_LSTM(input_dim)
    else:
        print("Invalid nn_type '%s'. Options are 'CNN' or 'LSTM'" % (nn_type))

def build_model_LSTM(input_dim):
    RNN = layers.LSTM
    start = time.time()

    model = Sequential()
    model.add(RNN(32, input_shape=input_dim))
    #model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)

    model.summary()
    return model

def build_model_CNN(input_dim):
    start = time.time()

    model = Sequential()
    model.add(Convolution1D(nb_filter=64, filter_length=2, input_shape=input_dim))
    model.add(Activation('relu'))
    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    print("> Compilation Time : ", time.time() - start)

    model.summary()
    return model

# Process input:
if input_type == "raw":
    print("Generating dataset from Raw signal")
    x_train, y_train = get_data(dftrain, SEQ_LEN)
    x_test, y_test = get_data(dftest, SEQ_LEN)

    x_train = np.reshape(x_train, x_train.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape + (1,))
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=42, shuffle=False)

elif input_type == "feat":
    print("Generating dataset from feature set")
    x_train_feat = dftrain[featnames].values
    x_test = dftest[featnames].values
    y_test = dftest["gt"].values
    scaler_feat = StandardScaler()
    scaler_feat.fit(x_train_feat)
    x_train_feat = scaler.transform(x_train_feat)
    x_test = scaler.transform(x_test)

    x_train_feat = np.reshape(x_train_feat, x_train_feat.shape + (1,))
    x_test = np.reshape(x_test, x_test.shape + (1,))

    x_train, x_val, y_train, y_val = train_test_split(x_train_feat, y_train, test_size=0.20, random_state=42, shuffle=False)
print("Done")


def run_experiment(input_type, epochs, batch_size, x_train, x_val, x_test, y_train, y_val, y_test):

    print("Training Shape: " ,x_train.shape[1:])

    if input_type == "raw":
        model = build_model(x_train.shape[1:], NN_TYPE)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_data=(x_val, y_val))
        print("Accuracy:", eval_acc(np.round(model.predict(x_test)), y_test))
        print("F1:", eval_f1(np.round(model.predict(x_test)), y_test))

    elif input_type == "feat":
        model = build_model(x_train.shape[1:], NN_TYPE)
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val))
        print("Accuracy:", eval_acc(np.round(model.predict(x_test)), y_test))
        print("F1:", eval_f1(np.round(model.predict(x_test)), y_test))

    return model

def load_trained_model(input_type):
	return

if TRAINING:
    print("Start training model...")
    model = run_experiment(input_type, epochs, batch_size, x_train, x_val, x_test, y_train, y_val, y_test)
    model.save(MODEL_OUTFILE)
    print("Model trained!")

else:
    model = load_model("models/" + MODEL_OUTFILE)
    print("Model loaded from disk!")

predictions = model.predict(x_test)

dftest["p_%s_%d" % (NN_TYPE, SEQ_LEN)] = predictions
dftest["%s_%d" % (NN_TYPE, SEQ_LEN)] = np.round(predictions)
dftest["gt_sleep_block"] = dftest["gt_sleep_block"].astype(int)
dftest["gt"] = dftest["gt"].astype(int)
dftest["actValue"] = dftest["actValue"].fillna(0.0).astype(int)
dftest[["mesaid","linetime","actValue","gt","gt_sleep_block","%s_%d" % (NN_TYPE, SEQ_LEN),"p_%s_%d" % (NN_TYPE, SEQ_LEN) ]].to_csv(RESULTS_OUTFILE, index=False)

print("Predictions made. Result save to %s." % (RESULTS_OUTFILE))
