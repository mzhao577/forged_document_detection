#!/bin/bash

pyScript="generate_charts_HTML_batch.py"
promptFile="./prompts/prompt3_generateHTMLFormat.txt"
#promptFile="./prompts/Prompt5_NoPageBreaker.txt"
outDir="./charts/Prompt3"
nChart=2
createPDF='True'

python $pyScript $promptFile $outDir $nChart $createPDF



