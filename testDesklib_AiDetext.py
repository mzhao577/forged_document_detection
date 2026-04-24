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

# Load model
model = DesklibAIDetectionModel.from_pretrained("desklib/ai-text-detector-v1.01")
tokenizer = AutoTokenizer.from_pretrained("desklib/ai-text-detector-v1.01")

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

# Read input file and output file from command-line arguments
if len(sys.argv) < 3:
    print("Usage: python testDesklib_AiDetext.py <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, 'r') as f:
    text = f.read()

filename = os.path.basename(input_file)
char_count = len(text)
word_count = len(text.split())

prediction, confidence = predict_single_text(text, model, tokenizer, device='cpu')

# Write header if output file doesn't exist yet
write_header = not os.path.exists(output_file)

with open(output_file, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["filename", "char_count", "word_count", "classification", "ai_probability"])
    writer.writerow([filename, char_count, word_count, prediction, f"{confidence:.4f}"])

print(f"Input file: {input_file}")
print(f"Prediction: {prediction} (AI probability: {confidence:.4f})")
print(f"Results appended to: {output_file}")
