from nltk.sem.evaluate import _ELEMENT_SPLIT_RE
from torch.utils.data import Dataset
import json
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import gensim.downloader as api
import nltk
import random
import clip


class MemesDataset(Dataset):
    def __init__(self, dataset_dir='hateful_memes', split='train', 
                 debug_small=False, crop=(0.5, 1), remove_stopwords=False,
                 split_text='space', text_model='glove',
                 drop_text=False):
        super().__init__()
        with open(f'{dataset_dir}/{split}.jsonl', 'r') as json_file:
            json_list = list(json_file)
        
        data = []
        for json_line in json_list:
            json_line = json.loads(json_line)
            data.append(json_line)
        if debug_small:
            data = list(np.random.choice(data, size=160, replace=False))

        self.data = data
        self.dataset_dir = dataset_dir
        self.split = split
        self.remove_stopwords = remove_stopwords
        self.split_text = split_text
        self.text_model = text_model
        self.drop_text = drop_text

        if split == 'train':
            self.img_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=(128, 128), scale=crop),
                #transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.img_transforms = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Load pre-trained word-vectors from gensim-data.
        if text_model == 'glove':
            word_vector_dict = api.load("glove-wiki-gigaword-300")
        elif text_model == 'twitter':
            word_vector_dict = api.load("glove-twitter-200")
        else:
            word_vector_dict = {}
        self.word_vector_dict = word_vector_dict

        if text_model == 'clip':
            text_features_dict = torch.load(f'data_prepare/{split}_text_features_dict.pt')
        else:
            text_features_dict = {}
        self.text_features_dict = text_features_dict

        # Load stopwords from nltk.
        stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords = stopwords

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]

        img_path = example['img']
        # Open an image.
        img = Image.open(f'{self.dataset_dir}/{img_path}').convert(mode='RGB')
        img_tensor = self.img_transforms(img)
        
        # Get the label.
        label = example['label']
        label = [label]
        label_tensor = torch.FloatTensor(label)

        # Get the text.
        if self.text_model == 'glove' or self.text_model == 'twitter':
            text = example['text']
            if self.split_text == 'space':
                text = [word for word in text.split()]
            elif self.split_text == 'tokenize':
                text = nltk.word_tokenize(text)
                text = [word for word in text if word.isalnum()]
                if self.drop_text:
                    if self.split == 'train':
                        random_size = int(round(0.8*len(text), 0))
                        text = random.sample(text, random_size)
            if self.remove_stopwords:
                text = [word for word in text if word not in self.stopwords]
            text_lst = []
            for word in text:
                if word in self.word_vector_dict:
                    word_vector = self.word_vector_dict[word]
                    text_lst.append(word_vector)

            if len(text_lst) == 0:
                if self.text_model == 'glove':
                    avg_word_vector = torch.zeros((300,)).float()
                elif self.text_model == 'twitter':
                    avg_word_vector = torch.zeros((200,)).float()
            else:
                text_lst = torch.from_numpy(np.array(text_lst)) # (N, 300/200)
                avg_word_vector = torch.mean(text_lst, dim=0) # (300/200,)

            out = {
            'img': img_tensor,
            'label': label_tensor,
            'text': avg_word_vector
            }
            
        elif self.text_model == 'clip':
            text_features = self.text_features_dict[index]
            out = {
            'img': img_tensor,
            'label': label_tensor,
            'text': text_features
            }

        return out