## Sleep Awake Benchmark

This code is part of the paper _''A Large Scale Benchmark to Validate Sleep-Wake Scoring Algorithms''_ currently under review.

### How To Use it

1. Download MESA datasets from sleepdata.org (https://sleepdata.org/datasets/mesa)
2. Process dataset with notebook -- generate_mesa_dataset_task.ipynb 
3. In general, all scripts will need to preprocess the data (i.e., get the mean activity value in a win of size X). To save time, we use implemented a preprocessing script (sleep_processdataset.py) which the only parameter is the task number. It outputs a file called hdf_task<taskid>. Run with:
```sh
$ python sleep_processdataset.py <taskid>
```
4. Run formulas (automatically reads dataset from hdf_task) and process all formulas:
```sh
$ python sleep_formulas.py <taskid>
```
5. Run machine learning models:
```sh
$ python sleep_ml.py <taskid>
```
6. Run LSTM/CNN:
```sh
$ python sleep_nn.py <taskid> <seq_len:20,50,100> <kind:LSTM,CNN>
```
7. Geting the results.
7.1. After running all scripts, a bunch of intermediary result files was created (e.g., task1_formulas.csv). The script _ship.sh_ collect them all and send them to their expected diretory (e.g., result or summary).
7.2. _sleep_summary.py_ generates the summaries files, which are the input for the ipython/jupyter notebooks (_result_analysis.ipynb_, _sleep_auc_plots.ipynb_ and _sleep_plot345.ipynb_). Notebooks provide an easy way to visualize and analyze the results, as well as generate the plots used in the manuscript.

