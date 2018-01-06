#!/usr/bin/env bash

SNLI_SOURCE=../snli_1.0
SNLI_DATA=snli
GLOVE=../en-fr.shuffled_random.300.vec

python process-snli.py --data_folder $SNLI_SOURCE --out_folder $SNLI_DATA

python preprocess-entail.py \
    --srcfile $SNLI_DATA/src-train.txt --targetfile $SNLI_DATA/targ-train.txt --labelfile $SNLI_DATA/label-train.txt \
    --srcvalfile $SNLI_DATA/src-dev.txt --targetvalfile $SNLI_DATA/targ-dev.txt --labelvalfile $SNLI_DATA/label-dev.txt \
    --srctestfile $SNLI_DATA/src-test.txt --targettestfile $SNLI_DATA/targ-test.txt --labeltestfile $SNLI_DATA/label-test.txt \
    --outputfile data/entail \
    --glove $GLOVE

python get_pretrain_vecs.py --glove $GLOVE --outputfile data/glove.hdf5 --dictionary data/entail.word.dict
