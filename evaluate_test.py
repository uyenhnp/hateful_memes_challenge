import torch
from torch.utils.data import DataLoader
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix

from models.text_image import TextImage3
from memes_dataset import MemesDataset
from utils import AverageMeter
from train import validate

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

# Load the model.
model = TextImage3(word_dim=512)
model.to(device)
checkpoint = torch.load('checkpoints/baseline/model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load test data.
test_dataset = MemesDataset(split='test_seen', text_model='clip')
print(f'# of examples: {len(test_dataset)}.')

def evaluate(model, dataset, device, num_workers):
    model.eval()
    batch_size = 50
    dataloader = DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers)
    test_acc_meter = AverageMeter()
    true_labels = []
    pred_labels = []
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
            # Get the predicted label.
            pred_label = (pred_prob > 0.5).long().float()
            pred_labels.append(pred_label.squeeze(1))
            # Calculate the accuracy.
            accuracy = (pred_label == label).long().sum()
            test_acc_meter.update(accuracy.item())
    test_accuracy = test_acc_meter.sum/len(test_dataset)
    return test_accuracy, pred_probs ,pred_labels, true_labels

# Evaluate on the test set.
test_accuracy, pred_probs ,pred_labels, true_labels = evaluate(model=model, dataset=test_dataset, 
    device=device, num_workers=4)
print(f'Test accuracy = {test_accuracy}.')

# Save y_preds into a file.
true_labels = torch.cat(true_labels, dim=0).cpu()
pred_labels = torch.cat(pred_labels, dim=0).cpu()
pred_probs = torch.cat(pred_probs, dim=0).cpu()
print(confusion_matrix(true_labels, pred_labels))

result = {'true_labels': true_labels, 'pred_labels': pred_labels, 'pred_probs': pred_probs}
result = pd.DataFrame(result)
result.to_csv('result.csv')
