

inputFile="./input/AIWritten_Sample5_ChinaAdvancementInRecentYears_WDuplicate.txt"
outDir="./output"
outPrefix="Sampletest5_falcon"

python binoculars_detection.py --output_result --input ${inputFile} --model_size falcon --output_prefix $outPrefix 


