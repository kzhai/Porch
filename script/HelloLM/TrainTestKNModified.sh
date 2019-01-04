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
        -kndiscount1 -kndiscount2 -kndiscount3 -kndiscount4 -kndiscount5 -kndiscount6 -kndiscount7 -kndiscount8 -kndiscount9 \
        -interpolate1 -interpolate2 -interpolate3 -interpolate4 -interpolate5 -interpolate6 -interpolate7 -interpolate8 -interpolate9 \
        -text $INPUT_DIRECTORY/train.txt \
        -lm $ARPA_FILE
        #> $OUTPUT_DIRECTORY/arpa,order=$j,kn=modified.out
        #2>$OUTPUT_DIRECTORY/arpa,order=$j,kn=modified.err
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
			#2>$OUTPUT_DIRECTORY/perplexity,order=$j,kn=modified.err
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
			#2>$OUTPUT_DIRECTORY/perplexity,order=$j,kn=modified.err
	fi

	tail -n 1 $PERPLEXITY_FILE
done