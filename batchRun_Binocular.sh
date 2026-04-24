
#pyScript="aitext_NewBinoculars_batch_v2.py"

#pyScript="aitext_OriginalBinocular.py"
pyScript="aitext_OriginalBinocular_localModels.py"
model="large"
inDir="./input"
outDir="./output"
outFile="AIHuman_Results1010_Binocular_${model}.csv"
threshold=0.91

python $pyScript  --input $inDir --model ${model}  --output_dir $outDir  --output_file $outFile --threshold $threshold 



: <<'COMMENT'
                                                                                                                                                                    
  python aitext_OriginalBinocular_localModels.py \                                                                                                                   
    --model small \
    --input_dir /path/to/text/files \                                                                                                                                
    --output_dir /path/to/output \                                                                                                                                 
    --output_file results.csv \                                                                                                                                      
    --threshold 0.9
                                                                                                                                                                     
  Arguments:                                                                                                                                                       

  ┌───────────────┬──────────┬──────────────────┬─────────────────────────────────────┐                                                                              
  │   Argument    │ Required │     Default      │             Description             │
  ├───────────────┼──────────┼──────────────────┼─────────────────────────────────────┤                                                                              
  │ --model       │ No       │ small            │ Model pair: small, large, or falcon │                                                                            
  ├───────────────┼──────────┼──────────────────┼─────────────────────────────────────┤
  │ --input_dir   │ Yes      │ —                │ Folder containing .txt files        │                                                                              
  ├───────────────┼──────────┼──────────────────┼─────────────────────────────────────┤                                                                              
  │ --output_dir  │ Yes      │ —                │ Directory for the output CSV        │                                                                              
  ├───────────────┼──────────┼──────────────────┼─────────────────────────────────────┤                                                                              
  │ --output_file │ No       │ results.csv      │ Output CSV filename                 │                                                                            
  ├───────────────┼──────────┼──────────────────┼─────────────────────────────────────┤                                                                              
  │ --threshold   │ No       │ Model's built-in │ Score threshold for AI detection    │
  └───────────────┴──────────┴──────────────────┴─────────────────────────────────────┘                                                                              
                                                                                                                                                                   
  Minimal example (uses defaults for model, output_file, and threshold):                                                                                             
   
  python aitext_OriginalBinocular_localModels.py \                                                                                                                   
    --input_dir ./input_texts \                                                                                                                                      
    --output_dir ./results

   
COMMENT
