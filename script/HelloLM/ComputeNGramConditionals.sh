#!/usr/bin/env bash

SRILM=/mnt/c/Users/kezhai/Workspace/Porch/HelloLM/srilm-i686-m64/

INPUT_DIRECTORY=$1
OUTPUT_DIRECTORY=$2

for j in 1 2 3 4 5 6 7 8 9; do
	echo "processing order=$j"

	NGRAM_FILE=$OUTPUT_DIRECTORY/gram=$j.txt
	if [ ! -f $NGRAM_FILE ]; then
		$SRILM/ngram-count \
			-order $j -unk \
			-text $INPUT_DIRECTORY/train.txt \
			-write$j $OUTPUT_DIRECTORY/gram=$j.txt
			#-no-eos -no-sos

		awk -F" " '{ print $NF, $0 }' $OUTPUT_DIRECTORY/gram=$j.txt | \
			sort -n -k1 -r | sed 's/^[0-9][0-9]* //' \
			> $OUTPUT_DIRECTORY/gram=$j.sort.txt
	fi
done

if [ $# -eq 2 ]; then
	exit 1
fi

MODEL_DIRECTORY=$3

for j in 2 3 4 5 6 7 8 9; do
	echo "processing order=$j"

	let k=$j-1
	NGRAM_FILE=$OUTPUT_DIRECTORY/gram=$j.sort.txt
    CONDITIONAL_FILE=$MODEL_DIRECTORY/train,prob=conditional,context=$k,order=$j,kn=modified.txt
    if [ ! -f $CONDITIONAL_FILE ]; then
        $SRILM/ngram \
            -lm $MODEL_DIRECTORY/order=$j,kn=modified.arpa \
            -order $j -unk -debug 2 \
            -counts $NGRAM_FILE \
            > $CONDITIONAL_FILE
    fi
done

for j in 2 3 4 5 6 7 8; do
	echo "processing order=$j"

	let k=$j-1
	let i=$j+1
	NGRAM_FILE=$OUTPUT_DIRECTORY/gram=$j.sort.txt
    CONDITIONAL_FILE=$MODEL_DIRECTORY/train,prob=conditional,context=$k,order=$i,kn=modified.txt
    if [ ! -f $CONDITIONAL_FILE ]; then
        $SRILM/ngram \
            -lm $MODEL_DIRECTORY/order=9,kn=modified.arpa \
            -order $i -unk -debug 2 \
            -counts $NGRAM_FILE \
            > $CONDITIONAL_FILE
    fi
done
