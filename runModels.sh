#!/bin/bash/

python hybrid_models.py 1024 cnn_lstm /data/seq2seq-data/
python hybrid_models.py 2048 cnn_lstm /data/seq2seq-data/
python hybrid_models.py 1024 cnnattn /data/seq2seq-data/
python hybrid_models.py 2048 cnnattn /data/seq2seq-data/

