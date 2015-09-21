spark-submit --master yarn-client --executor-memory 20G --driver-memory 2g --num-executors 2 --executor-cores 4 --files utils/utils.py GD/train.GD.accumulator.py

