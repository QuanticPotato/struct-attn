#!/usr/bin/env bash

# Train with structured attention

PATH_TO_DATA=data/entail
PATH_TO_WORD_VECS=data/glove.hdf5

th train-entail.lua -attn struct \
    -data_file $PATH_TO_DATA-train.hdf5 \
    -val_data_file $PATH_TO_DATA-val.hdf5 \
    -test_data_file $PATH_TO_DATA-test.hdf5 \
    -pre_word_vecs $PATH_TO_WORD_VECS \
    -savefile entail-struct
