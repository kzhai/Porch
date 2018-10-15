#from .ComputeConditionalProbabilities import cache_directory_name, output_directory_name, timestamp_prefix

import re

cache_directory_name = "data=train,cache=log_softmax(output)"
output_directory_name = "data=train,cache=softmax(output),context=history"
timestamp_prefix = "timestamp"

word_context_probability_pattern = re.compile(
	r'p\( (?P<word>.+?) \| (?P<context>.+?) \)\s+= \[(?P<ngram>\d+?)gram\] (?P<probability>.+?) \[ (?P<logprobinfo>.+?) \]')

# ngram_conditionals_pattern = re.compile(r'(?P<dataset>.+?),prob=conditional,context=(?P<context>\d+?),order=(?P<order>\d+?),kn=modified.txt')
# nlm_conditionals_pattern = re.compile(r'(?P<dataset>.+?),prob=conditional,context=(?P<context>\d+?).txt')
ngram_conditionals_pattern = re.compile(r'context=(?P<context>\d+?),order=(?P<order>\d+?).txt')
nlm_conditionals_pattern = re.compile(r'context=(?P<context>\d+?).txt')

ngram_sos = "<s>"
ngram_eos = "</s>"
nlm_eos = "<eos>"