from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import os

app = Flask(__name__)

# Load the model and tokenizer
model_path = os.path.join(os.getenv('APPDATA'), r'C:\inetpub\wwwroot\sentimentanalysis\SavedWeights.pt')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model.load_state_dict(torch.load(model_path))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the label mapping
label_mapping = {0: 'joy', 1: 'anger', 2: 'desire', 3: 'curiosity', 4: 'disappointment', 5: 'confusion'}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() 
    if 'text' not in data:
        return jsonify({'error': 'No data provided'}), 400
    
    try:
        texts = data['text']
        
        # Tokenize and encode the input texts
        inputs = [tokenizer.encode(text, add_special_tokens=True, max_length=256, pad_to_max_length=True, truncation=True) for text in texts]
        attention_masks = [[float(i > 0) for i in seq] for seq in inputs]
        
        # Convert to torch tensors
        inputs = torch.tensor(inputs)
        masks = torch.tensor(attention_masks)
        
        with torch.no_grad():
            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits
        
        # Convert logits to predicted labels
        predictions = np.argmax(logits.numpy(), axis=1)
        labeled_output = [label_mapping[label] for label in predictions]
        
        return jsonify({'answer': labeled_output}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400        

if __name__ == '__main__':
    # Set the WSGI application callable to allow using wfastcgi
    import sys
    sys.stdout = sys.stderr
    app.run()
