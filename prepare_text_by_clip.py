import torch
import clip
import json
import numpy as np

dataset_dir = 'hateful_memes'
train_file = 'train.jsonl'
dev_file = 'dev_seen.jsonl'
test_file = 'test_seen.jsonl'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the text data.
def load_text_data(dataset_dir='hateful_memes', split=train_file): 
    with open(f'{dataset_dir}/{split}', 'r') as json_file:
        json_list = list(json_file)

    data = []
    for i, json_line in enumerate(json_list):
        json_line = json.loads(json_line)
        img_text = json_line['text']
        data.append(img_text)
        if (i + 1) % 100 == 0:
            print(f'- Load {i+1} images.')
    
    return data

# Extract features using CLIP. 
def extract_features(data_lst, device=device, model=model):
    text_features_dict = {}
    for i, text in enumerate(data_lst):
        text = clip.tokenize(text, truncate=True).to(device) # (1, 77)
        with torch.no_grad():
            text_features = model.encode_text(text) # (1, 512)
        text_features_dict[i] = text_features
        if (i + 1) % 100 == 0:
            print(f'Done extracting {i+1} images.')
    return text_features_dict

# Save training text data. 
print('Load training text data:')
training_text_data = load_text_data(split=train_file)
print('Extract training text features:')
training_text_features_dict = extract_features(data_lst=training_text_data)
print(f'Save the training text features, len={len(training_text_features_dict)}:')
torch.save(training_text_features_dict, f'data_prepare/train_text_features_dict.pt')
print('Done.')

# Save dev text data. 
print('Load dev text data:')
dev_text_data = load_text_data(split=dev_file)
print('Extract dev text features:')
dev_text_features_dict = extract_features(data_lst=dev_text_data)
print(f'Save the dev text features, len={len(dev_text_features_dict)}:')
torch.save(dev_text_features_dict, f'data_prepare/dev_seen_text_features_dict.pt')
print('Done.')

# Save test text data. 
print('Load test text data:')
test_text_data = load_text_data(split=test_file)
print('Extract test text features:')
test_text_features_dict = extract_features(data_lst=test_text_data)
print(f'Save the test text features, len={len(test_text_features_dict)}:')
torch.save(test_text_features_dict, f'data_prepare/test_seen_text_features_dict.pt')
print('Done.')

# print(torch.load('data_prepare/training_text_features_dict.pt'))
# print(torch.load('data_prepare/dev_text_features_dict.pt'))
