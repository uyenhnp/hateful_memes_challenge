import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import argparse
from torch.utils.data import DataLoader
import time
import random
import os
import warnings
warnings.filterwarnings("ignore")

from models.baseline import Baseline, Baseline2, Baseline3, Baseline4, Baseline5
from models.text_image import TextImage, TextImage2, TextImage3
from memes_dataset import MemesDataset
from utils import AverageMeter, count_parameters

# wandb
import wandb
project = 'hateful_memes'
my_name = 'vkhoi'

def validate(model, dataloader, device):
    model.eval()
    dev_acc_meter = AverageMeter()
    with torch.no_grad():
        n = 0
        for iter, batch in enumerate(dataloader):
            for k in batch:
                batch[k] = batch[k].to(device)
            img = batch['img'].to(device)
            label = batch['label'].to(device)
            # Get the predicted label.
            y_pred = model(batch)
            # Calculate the batch accuracy.
            y_pred = (y_pred > 0.5).long().float()
            accuracy = (y_pred == label).long().sum()
            dev_acc_meter.update(accuracy.item())
            n += img.shape[0]
        print(f'Number examples in dev = {n}')
    return dev_acc_meter.sum/n
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--train_small', action='store_true') # default=False
    parser.add_argument('--n_workers', type=int, default=4) # default: use 1 main process
    parser.add_argument('--model', type=str, default='Baseline')
    parser.add_argument('--exp_name', type=str, default='Baseline')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--crop', type=float, default=0.5) 
    parser.add_argument('--stopwords', action='store_true')
    parser.add_argument('--split_text', type=str, default='space')
    parser.add_argument('--text_model', type=str, default='glove')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_feat_extract', type=float, default=1e-5)
    parser.add_argument('--drop_text', action='store_true')
    args = parser.parse_args()

    np.random.seed(24)
    torch.manual_seed(24)
    random.seed(24)

    if args.model == 'Baseline4' or args.model == 'Baseline5' or \
        args.model == 'TextImage' or args.model == 'TextImage2' or args.model == 'TextImage3':
        print(f'Train {args.model} with batch_size={args.batch_size},' 
          f' lr={args.learning_rate}, lr_feat_extract={args.lr_feat_extract},' 
          f' freeze={args.freeze}, weight_decay={args.weight_decay},'
          f' dropout={args.dropout}, crop={args.crop},'
          f' remove_stopwords={args.stopwords}, split_text={args.split_text},'
          f' text_model={args.text_model}, drop_text={args.drop_text}.')
    else:
        print(f'Train {args.model} with batch_size={args.batch_size}' 
          f' and learning_rate={args.learning_rate}.')

    config = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'n_epochs': args.n_epochs,
        'model': args.model,
        'dropout': args.dropout,
        'crop': args.crop,
        'stopwords': args.stopwords,
        'split_text': args.split_text,
        'text_model': args.text_model
    }

    wandb.init(
        project=project,
        name=args.exp_name,
        config=config,
        entity=my_name
    )

    wandb.define_metric("train/epoch")
    wandb.define_metric("train/*", step_metric="train/epoch")
    wandb.define_metric("dev/*", step_metric="train/epoch")

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using {device}')

    # 1. Prepare data.
    dataset = MemesDataset(
        split='train', 
        debug_small=args.train_small, 
        crop=(args.crop, 1), 
        remove_stopwords=args.stopwords,
        split_text=args.split_text, 
        text_model=args.text_model,
        drop_text=args.drop_text, 
        )
    print(f"# of examples: {len(dataset)}")
    batch_size = args.batch_size
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=args.n_workers)
    print(f"# batch: {len(dataloader)}")
    
    dev_dataset = MemesDataset(
        split='dev_seen',
        remove_stopwords=args.stopwords,
        split_text=args.split_text,
        text_model=args.text_model,
        )
    print(f"# of dev examples: {len(dev_dataset)}")
    dev_dataloader = DataLoader(
        dev_dataset, 
        batch_size=50, 
        shuffle=False, 
        num_workers=args.n_workers)
    print(f"# of dev batch: {len(dev_dataloader)}")

    # 2. Create a model.
    if args.model == 'Baseline': 
        model = Baseline()
    elif args.model == 'Baseline2':
        model = Baseline2()
    elif args.model == 'Baseline3':
        model = Baseline3()
    elif args.model == 'Baseline4':
        model = Baseline4(
            freeze_feat_extractor=args.freeze,
            p=args.dropout)
    elif args.model == 'Baseline5':
        model = Baseline5(
            freeze_feat_extractor=args.freeze,
            p=args.dropout)
    elif args.model == 'TextImage':
        if args.text_model == 'glove':
            word_dim = 300
        elif args.text_model == 'twitter':
            word_dim = 200
        elif args.text_model == 'clip':
            word_dim = 512
        model = TextImage(
            freeze_feat_extractor=args.freeze,
            p=args.dropout,
            word_dim=word_dim
        )
    elif args.model == 'TextImage2':
        if args.text_model == 'glove':
            word_dim = 300
        elif args.text_model == 'twitter':
            word_dim = 200
        elif args.text_model == 'clip':
            word_dim = 512
        model = TextImage2(
            freeze_feat_extractor=args.freeze,
            p=args.dropout,
            word_dim=word_dim
        )
    elif args.model == 'TextImage3':
        word_dim = 512
        model = TextImage3(
            freeze_feat_extractor=args.freeze,
            p=args.dropout,
            word_dim=word_dim
        )

    model.to(device)
    total_params = count_parameters(model)
    wandb.run.summary["num_parameters"] = total_params

    # 3. Lost function.
    loss_fn = torch.nn.BCELoss(reduction='mean')

    # 4. Optimizer.
    learning_rate = args.learning_rate
    feat_extractor_params = [
        param
        for name, param in model.named_parameters()
        if 'feat_extractor' in name and param.requires_grad
    ]
    optimizer = torch.optim.Adam([
        {'params': feat_extractor_params, 'lr': args.lr_feat_extract},
        {'params': model.fc.parameters()}
        ], lr=learning_rate, weight_decay=args.weight_decay)

    # 5. Train the model.
    epochs = args.n_epochs
    best_dev_accuracy = 0
    for epoch in range(epochs):
        model.train()
        if args.freeze:
            model.feat_extractor.eval()

        loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        # Time check.
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end_time = time.time()
        
        for iter, batch in enumerate(dataloader):
            data_time.update(time.time() - end_time)

            for k in batch:
                batch[k] = batch[k].to(device)
            label = batch['label']
            img = batch['label']

            # Forward pass: calculate y_pred.
            y_pred = model(batch)
            # Calculate the loss.
            loss = loss_fn(y_pred, label)
            loss_meter.update(loss.item())
            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()
            # Backward pass: compute gradient of the loss with respect to all parameters.
            loss.backward()
            # Update the weights using gradient descent.
            optimizer.step()
            # Calculate accuracy.
            y_pred = (y_pred > 0.5).long().float()
            accuracy = (y_pred == label).long().sum()/img.shape[0]
            train_acc_meter.update(accuracy.item())

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (iter + 1) % 50 == 0:
                print(f'Iter {iter+1}/{len(dataloader)}: Acc = {train_acc_meter.avg:.4f}'
                      f', Batch_time = {batch_time.avg:.2f}, Data_time = {data_time.avg:.2f}')
        
        # Compute dev accuracy.
        dev_accuracy = validate(model, dev_dataloader, device)
        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            # Create checkpoint path.
            ckpt_path = f'checkpoints/{args.model}/{args.exp_name}'
            if not os.path.exists(ckpt_path):
                os.makedirs(ckpt_path)
            # Save the checkpoint.
            file_path = f'{ckpt_path}/model.pt'

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, file_path)
            print(f'Save checkpoint at epoch {epoch}.')
        
        print(f'epoch={epoch+1}, loss={loss_meter.avg:.4f}, train_accuracy={train_acc_meter.avg:.4f},'
              f' dev_accuracy={dev_accuracy:.4f}')

        # Plot on wandb.
        wandb.log({
            'train/loss': loss_meter.avg,
            'train/acc': train_acc_meter.avg,
            'dev/acc': dev_accuracy,
            'train/epoch': epoch + 1
        }, commit=True)


if __name__ == '__main__':
    main()