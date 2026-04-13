

python aitext_NewBinoculars_batch_v2.py  --model large --input input --output_dir output --output_file  max5_binocular_large.csv


: <<'COMMENT'

 Here's how to call the script:                                                                                                                                                  
  Basic usage (uses falcon model by default):                                                                                                                                     
  python aitext_NewBinoculars_batch_v2.py --input /path/to/txt/folder                                                                                                             
                                                                                                                                                                                  
  With a specific model (falcon, small, or large):                                                                                                                                
  python aitext_NewBinoculars_batch_v2.py --input /path/to/txt/folder --model large                                                                                               
                                                                                                                                                                                  
  With custom output location:                                                                                                                                                    
  python aitext_NewBinoculars_batch_v2.py --input /path/to/txt/folder --output_dir /path/to/output --output_file results.csv                                                    
                                                                                                                                                                                  
  Arguments:                                                                                                                                                                      
   
  ┌───────────────┬──────────┬─────────────────────────────┬───────────────────────────────────────────────────────────────────────┐                                              
  │   Argument    │ Required │           Default           │                              Description                              │                                            
  ├───────────────┼──────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────────┤                                              
  │ --input       │ Yes      │ —                           │ Folder containing .txt files to analyze                               │                                            
  ├───────────────┼──────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │ --model       │ No       │ falcon                      │ Model pair: falcon, small (gpt2→gpt2-medium), large (gpt2→gpt2-large) │                                              
  ├───────────────┼──────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────────┤                                              
  │ --output_dir  │ No       │ same as input               │ Directory for the output CSV                                          │                                              
  ├───────────────┼──────────┼─────────────────────────────┼───────────────────────────────────────────────────────────────────────┤                                              
  │ --output_file │ No       │ binoculars_batch_result.csv │ Output CSV filename                                                   │                                            
  └───────────────┴──────────┴─────────────────────────────┴───────────────────────────────────────────────────────────────────────┘                                              
   
  The script processes all .txt files in the input folder and writes results to a single CSV with columns: filename, word_count, char_count, model, binoculars_score, threshold,  
  classification, ai_probability, human_probability.                                                                                                                            
                                                                                                                                                                                  
  Also note: the --model help text still references "medium" — want me to update that?                                                                                            
   
COMMENT
