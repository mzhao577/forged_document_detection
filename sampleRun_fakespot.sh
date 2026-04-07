

inputFile="./input/AIWritten_Sample5_ChinaAdvancementInRecentYears_WDuplicate.txt"
outDir="./output"
outPrefix="Sampletest5_fakespot"

python fakespot_detection.py  --output_result --input $inputFile --output_dir $outDir --output_prefix $outPrefix


