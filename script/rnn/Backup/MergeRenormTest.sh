#!/usr/bin/env bash

SRILM=/mnt/c/Users/kezhai/Workspace/Porch/HelloLM/srilm-i686-m64/

DATA_DIRECTORY=$1
INPUT_DIRECTORY=$2

#mkdir $INPUT_DIRECTORY/approximation=1-9
#mv $INPUT_DIRECTORY/data=test $INPUT_DIRECTORY/approximation=1-9/
#mv $INPUT_DIRECTORY/data=train $INPUT_DIRECTORY/approximation=1-9/
#mv $INPUT_DIRECTORY/original.arpa $INPUT_DIRECTORY/approximation=1-9/
#mv $INPUT_DIRECTORY/renormed.arpa $INPUT_DIRECTORY/approximation=1-9/

NNLM_DIRECTORY=$INPUT_DIRECTORY/
NGRAM_DIRECTORY=./data/ptb/kn=modified,data=raw/breakdown

for i in 1; do
	#let itemp=$i-1
	MODEL_DIRECTORY=$INPUT_DIRECTORY/approximation=$i-9
	mkdir $MODEL_DIRECTORY

	ORIGINAL_ARPA_FILE=$MODEL_DIRECTORY/original.arpa
	if [ ! -f $ORIGINAL_ARPA_FILE ]; then
		python3 -u script/rnn/MergeNGrams/MergeNGrams.py \
			--primary_ngram_directory=$NNLM_DIRECTORY \
			--secondary_ngram_directory=$NGRAM_DIRECTORY \
			--output_file=$ORIGINAL_ARPA_FILE \
			--index=$i
	fi

	RENORMED_ARPA_FILE=$MODEL_DIRECTORY/renormed.arpa
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

    TEST_PPLX_DIRECTORY=$MODEL_DIRECTORY/pplx=test/
    TRAIN_PPLX_DIRECTORY=$MODEL_DIRECTORY/pplx=train/
    mkdir $TEST_PPLX_DIRECTORY
    mkdir $TRAIN_PPLX_DIRECTORY

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

        tail -n 1 $PERPLEXITY_FILE

        PERPLEXITY_FILE=$TRAIN_PPLX_DIRECTORY/order=$j.pplx
        if [ ! -f $PERPLEXITY_FILE ]; then
            $SRILM/ngram \
                -lm $RENORMED_ARPA_FILE \
                -order $j \
                -unk -debug 2 \
                -ppl $DATA_DIRECTORY/train.txt \
                > $PERPLEXITY_FILE
                #2>$OUTPUT_DIRECTORY/perplexity,order=$j,wb.err
                #-no-sos -no-eos \
        fi

        tail -n 1 $PERPLEXITY_FILE
    done                                                                    
done                                                                                        