#!/usr/bin/env bash

SRILM=/mnt/c/Users/kezhai/Workspace/Porch/script/HelloLM/srilm-i686-m64/

DATA_DIRECTORY=$1
INPUT_DIRECTORY=$2

#NNLM_DIRECTORY=$INPUT_DIRECTORY/breakdown
#NGRAM_DIRECTORY=./data/ptb/kn=modified,data=raw/breakdown

ORIGINAL_ARPA_FILE=$INPUT_DIRECTORY/original.arpa
if [ ! -f $ORIGINAL_ARPA_FILE ]; then
    python3 -u script/rnn/MergeNGrams/MergeNGrams.py \
        --primary_ngram_directory=$INPUT_DIRECTORY \
        --output_file=$ORIGINAL_ARPA_FILE
fi

RENORMED_ARPA_FILE=$INPUT_DIRECTORY/renormed.arpa
if [ ! -f $RENORMED_ARPA_FILE ]; then
    $SRILM/ngram \
        -lm $ORIGINAL_ARPA_FILE \
        -order 9 \
        -unk \
        -debug 2 \
        -renorm \
        -write-lm $RENORMED_ARPA_FILE
        #> $OUTPUT_DIRECTORY/arpa,order=$j,wb.out
        #2>$OUTPUT_DIRECTORY/arpa,order=$j,wb.err
        #-no-sos -no-eos \
fi

TEST_PPLX_DIRECTORY=$INPUT_DIRECTORY/pplx=test/
mkdir $TEST_PPLX_DIRECTORY
VALIDATE_PPLX_DIRECTORY=$INPUT_DIRECTORY/pplx=validate/
mkdir $VALIDATE_PPLX_DIRECTORY
#TRAIN_PPLX_DIRECTORY=$INPUT_DIRECTORY/pplx=train/
#mkdir $TRAIN_PPLX_DIRECTORY

for j in 1 2 3 4 5 6 7 8 9; do
    echo "processing order=$j"

    PERPLEXITY_FILE=$TEST_PPLX_DIRECTORY/order=$j.pplx
    if [ ! -f $PERPLEXITY_FILE ]; then
        $SRILM/ngram \
            -lm $RENORMED_ARPA_FILE \
            -order $j \
            -unk -debug 2 \
            -ppl $DATA_DIRECTORY/test.txt \
            > $PERPLEXITY_FILE
            #2>$OUTPUT_DIRECTORY/perplexity,order=$j,wb.err
            #-no-sos -no-eos \
    fi

#    tail -n 1 $PERPLEXITY_FILE

    PERPLEXITY_FILE=$VALIDATE_PPLX_DIRECTORY/order=$j.pplx
    if [ ! -f $PERPLEXITY_FILE ]; then
        $SRILM/ngram \
            -lm $RENORMED_ARPA_FILE \
            -order $j \
            -unk -debug 2 \
            -ppl $DATA_DIRECTORY/validate.txt \
            > $PERPLEXITY_FILE
            #2>$OUTPUT_DIRECTORY/perplexity,order=$j,wb.err
            #-no-sos -no-eos \
    fi

#    tail -n 1 $PERPLEXITY_FILE

#    PERPLEXITY_FILE=$TRAIN_PPLX_DIRECTORY/order=$j.pplx
#    if [ ! -f $PERPLEXITY_FILE ]; then
#        $SRILM/ngram \
#            -lm $RENORMED_ARPA_FILE \
#            -order $j \
#            -unk -debug 2 \
#            -ppl $DATA_DIRECTORY/train.txt \
#            > $PERPLEXITY_FILE
#            #2>$OUTPUT_DIRECTORY/perplexity,order=$j,wb.err
#            #-no-sos -no-eos \
#    fi

#    tail -n 1 $PERPLEXITY_FILE
done                                                                    

for j in 1 2 3 4 5 6 7 8 9; do
    PERPLEXITY_FILE=$TEST_PPLX_DIRECTORY/order=$j.pplx
    tail -n 1 $PERPLEXITY_FILE
done