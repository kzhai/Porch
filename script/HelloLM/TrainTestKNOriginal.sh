#!/usr/bin/env bash

SRILM=/mnt/c/Users/kezhai/Workspace/Porch/HelloLM/srilm-i686-m64/

INPUT_DIRECTORY=$1
OUTPUT_DIRECTORY=$2

mkdir $OUTPUT_DIRECTORY
#mkdir $OUTPUT_DIRECTORY/wb

ARPA_FILE=$OUTPUT_DIRECTORY/ngram=9.arpa
if [ ! -f $ARPA_FILE ]; then
    $SRILM/ngram-count \
		-order 9 \
		-unk \
		-gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0 -gt5min 0 -gt6min 0 -gt7min 0 -gt8min 0 -gt9min 0 \
		-ukndiscount1 -ukndiscount2 -ukndiscount3 -ukndiscount4 -ukndiscount5 -ukndiscount6 -ukndiscount7 -ukndiscount8 -ukndiscount9 \
		-interpolate1 -interpolate2 -interpolate3 -interpolate4 -interpolate5 -interpolate6 -interpolate7 -interpolate8 -interpolate9 \
		-text $INPUT_DIRECTORY/train.txt \
		-lm $ARPA_FILE
fi

mkdir $OUTPUT_DIRECTORY/pplx=test
mkdir $OUTPUT_DIRECTORY/pplx=train

for j in 1 2 3 4 5 6 7 8 9; do
	echo "processing order=$j"

	PERPLEXITY_FILE=$OUTPUT_DIRECTORY/pplx=test/order=$j.pplx
	if [ ! -f $PERPLEXITY_FILE ]; then
		$SRILM/ngram \
			-lm $ARPA_FILE \
			-order $j \
			-unk -debug 2 \
			-ppl $INPUT_DIRECTORY/test.txt \
			> $PERPLEXITY_FILE
	fi

	tail -n 1 $PERPLEXITY_FILE

	PERPLEXITY_FILE=$OUTPUT_DIRECTORY/pplx=train/order=$j.pplx
	if [ ! -f $PERPLEXITY_FILE ]; then
		$SRILM/ngram \
			-lm $ARPA_FILE \
			-order $j \
			-unk -debug 2 \
			-ppl $INPUT_DIRECTORY/train.txt \
			> $PERPLEXITY_FILE
	fi

	tail -n 1 $PERPLEXITY_FILE
done