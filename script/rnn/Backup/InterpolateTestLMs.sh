#!/usr/bin/env bash

SRILM=/mnt/c/Users/kezhai/Workspace/Porch/HelloLM/srilm-i686-m64/

DATA_DIRECTORY=$1
MKN_MODEL=$2
NNLM_MODEL=$3
OUTPUT_DIRECTORY=$4

mkdir $OUTPUT_DIRECTORY

for LAMBDA in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 ; do
    for ORDER in 1 2 3 4 5 6 7 8 9 ; do
        echo "processing order=$ORDER"

        PERPLEXITY_FILE=$OUTPUT_DIRECTORY/mkn=$LAMBDA,order=$ORDER.pplx
        if [ ! -f $PERPLEXITY_FILE ]; then
            $SRILM/ngram \
                -lm $MKN_MODEL \
                -mix-lm $NNLM_MODEL \
                -lambda $LAMBDA \
                -order $ORDER \
                -unk -debug 2 \
                -ppl $DATA_DIRECTORY/test.txt \
                > $PERPLEXITY_FILE
                #2>$OUTPUT_DIRECTORY/perplexity,order=$j,wb.err
                #-no-sos -no-eos \
        fi

        tail -n 1 $PERPLEXITY_FILE
    done                                                                    
done