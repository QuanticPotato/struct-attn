#!/bin/bash

echo "Launching ..."

LANGUAGE=fr

PATH_TO_DATA=data/entail
PATH_TO_WORD_VECS=data/glove-$LANGUAGE.hdf5


th evaluate-entail.lua -attn none -pre_word_vecs $PATH_TO_WORD_VECS -gpuid 1
