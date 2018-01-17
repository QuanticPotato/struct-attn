#!/usr/bin/env bash

# Arguments : $1 = language
# You must run preprocess before !!

LANGUAGE=$1

SNLI_DATA=snli
GLOVE=../en-fr.$LANGUAGE.shuffled_random.300.vec

python get_pretrain_vecs.py --glove $GLOVE --outputfile data/glove-$LANGUAGE.hdf5 --dictionary data/entail.word.dict
