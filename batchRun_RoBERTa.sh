⏺ Usage:

  # Analyze a single file (uses OpenAI RoBERTa model by default)
  python aitext_detectionbyRobertA_batch.py --input input/testfiles/HumanWritten_Sample1_ParisTravelIntro.txt

  # Analyze all .txt files in a folder
  python aitext_detectionbyRobertA_batch.py --input input/testfiles/

  # Use the Fakespot model instead
  python aitext_detectionbyRobertA_batch.py --input input/testfiles/ --model fakespot

  # Specify output directory and filename
  python aitext_detectionbyRobertA_batch.py --input input/testfiles/ --output_dir output --output_file my_results.csv

  Options:

  ┌───────────────┬─────────────────────────────────┬─────────────────┐
  │     Flag      │           Description           │     Default     │
  ├───────────────┼─────────────────────────────────┼─────────────────┤
  │ --input       │ Input file or folder (required) │ —               │
  ├───────────────┼─────────────────────────────────┼─────────────────┤
  │ --model, -m   │ openai or fakespot              │ openai          │
  ├───────────────┼─────────────────────────────────┼─────────────────┤
  │ --output_dir  │ Directory for output CSV        │ . (current dir) │
  ├───────────────┼─────────────────────────────────┼─────────────────┤
  │ --output_file │ Output CSV filename             │ auto-generated  │
  └───────────────┴─────────────────────────────────┴─────────────────┘

  Note: The models are loaded from local cache paths (~/.cache/huggingface/hub/), so they need to be downloaded first.

