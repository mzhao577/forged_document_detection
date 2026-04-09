⏺ Usage:

  # Basic: analyze a file (auto-detects best local model)
  python aitext_detectionbyBinoculars_batch.py --input input/testfiles/HumanWritten_Sample1_ParisTravelIntro.txt

  # Analyze a folder of .txt files
  python aitext_detectionbyBinoculars_batch.py --input input/testfiles/

  # Use Hugging Face API (no local GPU/memory needed, requires HF token)
  python aitext_detectionbyBinoculars_batch.py --input input/testfiles/ --use_api --hf_token YOUR_TOKEN

  # Force a specific local model size
  python aitext_detectionbyBinoculars_batch.py --input input/testfiles/ --model_size medium

  # Use offline mode (locally cached models only)
  python aitext_detectionbyBinoculars_batch.py --input input/testfiles/ --model_size falcon --offline

  # Custom output location and prefix
  python aitext_detectionbyBinoculars_batch.py --input input/testfiles/ --output_dir output --output_prefix mytest

  Options:

  ┌─────────────────┬───────────────────────────────────────────────────────────────────┬────────────────┐
  │      Flag       │                            Description                            │    Default     │
  ├─────────────────┼───────────────────────────────────────────────────────────────────┼────────────────┤
  │ --input         │ Input file or folder (required)                                   │ —              │
  ├─────────────────┼───────────────────────────────────────────────────────────────────┼────────────────┤
  │ --model_size    │ auto, falcon (~28GB), large (~4GB), medium (~2GB), small (~500MB) │ auto           │
  ├─────────────────┼───────────────────────────────────────────────────────────────────┼────────────────┤
  │ --use_api       │ Use HF Inference API instead of local models                      │ off            │
  ├─────────────────┼───────────────────────────────────────────────────────────────────┼────────────────┤
  │ --hf_token      │ HF API token (or set HF_TOKEN env var)                            │ —              │
  ├─────────────────┼───────────────────────────────────────────────────────────────────┼────────────────┤
  │ --offline       │ Use only locally cached models                                    │ off            │
  ├─────────────────┼───────────────────────────────────────────────────────────────────┼────────────────┤
  │ --threshold     │ Detection threshold (lower = more likely AI)                      │ auto           │
  ├─────────────────┼───────────────────────────────────────────────────────────────────┼────────────────┤
  │ --output_dir    │ Output directory                                                  │ same as input  │
  ├─────────────────┼───────────────────────────────────────────────────────────────────┼────────────────┤
  │ --output_file   │ Output CSV filename                                               │ auto-generated │
  ├─────────────────┼───────────────────────────────────────────────────────────────────┼────────────────┤
  │ --output_prefix │ Prefix for auto-generated filename                                │ binoculars     │
  └─────────────────┴───────────────────────────────────────────────────────────────────┴────────────────┘

  API setup (if using --use_api):
  1. Create a free account at huggingface.co
  2. Get a token at huggingface.co/settings/tokens
  3. Either export HF_TOKEN=your_token or pass --hf_token


