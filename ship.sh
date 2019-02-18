
for f in task1_formulas.csv task2_formulas.csv task1_ml.csv task2_ml.csv; do
    if [ -f $f ]; then
        echo "Zipping and moving $f to results";
        gzip $f;
        mv ${f}.gz results/;
    fi
done

for TASK in "1" "2" ; do
    for NNMODEL in "CNN" "LSTM"; do
        for SIZE in "20" "50" "100"; do
            f="task${TASK}_${NNMODEL}_raw_${SIZE}.csv"
            if [ -f $f ]; then
                echo "Zipping and moving $f to results";
                gzip $f;
                mv ${f}.gz results/;
            fi
        done
    done
done

for TASK in "1" "2" ; do
    for NNMODEL in "CNN" "LSTM"; do
        for SIZE in "20" "50" "100"; do
            f="model_${NNMODEL}_task${TASK}_raw_seq${SIZE}.pkl"
            if [ -f $f ]; then
                echo "Just moving $f to models";
                mv ${f} models/;
            fi
        done
    done
    
    f="model_ml_task${TASK}.pkl"
    if [ -f $f ]; then
        echo "Just moving $f to models";
        mv ${f} models/;
    fi
done


for f in task1_summary.csv task2_summary.csv; do
    if [ -f $f ]; then
        echo "Just moving $f to summaries";
        mv ${f} summaries/;
    fi
done

for f in task1_results.pkl task2_results.pkl; do
    if [ -f $f ]; then
        echo "Just moving $f to results";
        mv ${f} results/;
    fi
done

