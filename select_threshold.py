import torch
import numpy as np
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix

from models.text_image import TextImage3
from memes_dataset import MemesDataset
from utils import AverageMeter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

# Load the model.
model = TextImage3(word_dim=512)
model.to(device)
checkpoint = torch.load('checkpoints/baseline/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load dev data.
dev_dataset = MemesDataset(split='dev_seen', text_model='clip')
print(f'# of examples: {len(dev_dataset)}.')

dev_dataloader = DataLoader(
        dev_dataset, 
        batch_size=50, 
        shuffle=False, 
        num_workers=4)
print(f'# of batch: {len(dev_dataloader)}.')

# Function to return predicted probabilities. 
def predict_prob(model, dataloader, device):
    model.eval()
    true_labels = []
    pred_probs = []
    with torch.no_grad():
        for iter, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)
            img = batch['img'].to(device)
            label = batch['label'].to(device)
            true_labels.append(label.squeeze(1))
            # Get the predicted prob.
            pred_prob = model(batch)
            pred_probs.append(pred_prob.squeeze(1))
    true_labels = torch.cat(true_labels, dim=0).cpu()
    pred_probs = torch.cat(pred_probs, dim=0).cpu()
    return pred_probs, true_labels

# Get predicted probs.
pred_probs, true_labels = predict_prob(
    model=model, 
    dataloader=dev_dataloader, 
    device=device)

# Get predicted labels.
def predict_labels(threshold=0.5, pred_probs=pred_probs, true_labels=true_labels):
    pred_labels = (pred_probs > threshold).long().float()
    n_correct_pairs = (pred_labels == true_labels).long().sum()
    accuracy = n_correct_pairs / len(true_labels)
    return pred_labels, accuracy

# Choose the optimal threshold
thresholds = np.arange(0.0, 1.05, 0.05)
best_accuracy = 0
best_thres = -0.1
best_pred_labels = []
for thres in thresholds:
    pred_labels, accuracy = predict_labels(threshold=thres)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_thres = thres
        best_pred_labels = pred_labels
    print(f'Threshold = {thres:.2f}, Development Accuracy = {accuracy:.3f}')
print(f'Optimal Threshold = {best_thres }, Development Accuracy = {best_accuracy:.3f}')
print(confusion_matrix(true_labels, best_pred_labels))