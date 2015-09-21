#output="hdfs://hqz-ubuntu-master:9000/data/feat_stat.3"
#hadoop fs -rmr ${output}
#spark-submit --master yarn-client --executor-memory 12G --driver-memory 2g --executor-cores 2 --driver-java-options "-Xms10g -Xmx10g" feat_stat_2.py
spark-submit --master yarn-client --executor-memory 20G --driver-memory 2g --num-executors 2 --executor-cores 4 --py-files utils.py train_LBFGS_broadcast.py
#train.py
#spark-submit --master local[4] --executor-memory 18G feat_stat.py

