import sys
import os
import csv
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig
    _tied_weights_keys = []
    all_tied_weights_keys = {}

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Classifier
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())
        
        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        
        return output

# Load model from local cache
LOCAL_MODEL_PATH = os.path.expanduser("~/.cache/huggingface/hub/models--desklib--ai-text-detector-v1.01/snapshots")
# Use the latest snapshot
snapshot_dir = os.path.join(LOCAL_MODEL_PATH, os.listdir(LOCAL_MODEL_PATH)[0])
model = DesklibAIDetectionModel.from_pretrained(snapshot_dir)
tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)

# Predict
def predict_single_text(text, model, tokenizer, device, max_len=768, threshold=0.5):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        prob = torch.sigmoid(logits).item()
    
    prediction = "AI_Written" if prob > threshold else "Human_Written"
    return prediction, prob

# Read input dir, output file, and threshold from command-line arguments
if len(sys.argv) < 4:
    print("Usage: python aitext_desklib_batch.py <input_dir> <output_file> <threshold>")
    sys.exit(1)

input_dir = sys.argv[1]
output_file = sys.argv[2]
threshold = float(sys.argv[3])

# Get all text files in input_dir, sorted by name
files = sorted([f for f in os.listdir(input_dir)
                if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.txt')])

processed = 0
skipped = 0

with open(output_file, 'w', newline='') as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["filename", "char_count", "word_count", "TrueLabel", "Classification", "ai_probability", "threshold", "Prediction", "model"])

    for filename in files:
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        except Exception as e:
            print(f"SKIPPED {filename}: {e}")
            skipped += 1
            continue

        char_count = len(text)
        word_count = len(text.split())
        TrueLabel = "AI_Written" if filename.startswith("AI") else "Human_Written"

        classification, confidence = predict_single_text(text, model, tokenizer, device='cpu')
        prediction = "AI_Written" if confidence > threshold else "Human_Written"

        writer.writerow([filename, char_count, word_count, TrueLabel, classification, f"{confidence:.4f}", threshold, prediction, "desklib"])
        print(f"{filename} -> Prediction: {prediction} (AI prob: {confidence:.4f}, threshold: {threshold})")
        processed += 1

print(f"\nResults written to: {output_file}")
print(f"Total files processed: {processed}, skipped: {skipped}")
