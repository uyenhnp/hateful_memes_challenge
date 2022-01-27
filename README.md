## HATEFUL MEMES CHALLENGE

### 1. Introduction
Meme has become an essential element of the current internet era. Meme containing hateful content is without a doubt also becoming more and more widespread on many social media sites. Therefore, having a machine learning system that can automatically detect memes with hateful content is the goal of these social media sites, especially Facebook.

In this project, we aim to use the [Hateful Memes dataset](https://hatefulmemeschallenge.com/#about) provided by Facebook AI to build a machine learning model that can detect hateful speech in memes. When we view a meme, it is impossible to understand its true meaning by looking separately at only its image or text. It is necessary to combine the information in both modalities to comprehend the meme's true interpretation. Therefore, the purpose of this work is to explore multiple approaches to implement a multi-modal convolution neural network that can classify a meme as "hateful" or "not hateful".

![Example](https://github.com/uyenhnp/hateful_memes_challenge/blob/master/theme.jpg)

### 2. Data
The official web page for the dataset: [link](https://hatefulmemeschallenge.com/#about)

This dataset contains pairs of text & image with the following statistics:

| Dataset | Number of samples |
| ----------- | ----------- |
| Training set | 8500 |
| Development set | 500 |
| Test set | 1000 |

### 3. Models
I performed two types of models: Unimodal and Multimodal. While the unimodal models use only the image features, the multimodal models combine both the image and text features to detect hate speech in memes.

**Model: Baseline Unimodal 1**
![Example](https://github.com/uyenhnp/hateful_memes_challenge/blob/master/model_architecture_images/baseline_unimodal1.png)

**Model: Baseline Unimodal 2**
![Example](https://github.com/uyenhnp/hateful_memes_challenge/blob/master/model_architecture_images/baseline_unimodal2.png)

**Model: Baseline Multimodal 1**
![Example](https://github.com/uyenhnp/hateful_memes_challenge/blob/master/model_architecture_images/baseline_multimodal1.png)

**Model: Baseline Multimodal 2**
![Example](https://github.com/uyenhnp/hateful_memes_challenge/blob/master/model_architecture_images/baseline_multimodal2.png)

**Model: Text Image Multimodal**
![Example](https://github.com/uyenhnp/hateful_memes_challenge/blob/master/model_architecture_images/textimage.png)

| Model | Description | Development Accuracy |
| ----------- | ----------- | ----------- |
| Baseline Unimodal 1 | Designed CNN | 0.506 |
| Baseline Unimodal 2 | ResNet18 | 0.524 | 
| Baseline Multimodal 1| ResNet18 & Glove-twitter-200 | 0.574 | 
| Baseline Multimodal 2 | ResNet50 & Glove-twitter-200 | 0.576 | 
| **Text Image Multimodal** | **ResNet50 & CLIP** | **0.616** | 

**The best model is Text Image Multimodal, which has a test accuracy of 0.637.**

### 4. Usage
To use Text Image Multimodal, please folow the code below. 

#### Reproduce the model
If you want to reproduce the model, please download the [dataset](https://hatefulmemeschallenge.com/#about) and put the images into the folder `hateful_memes/img` (the location of the images should be as follows: `hateful_memes/img/{img_name}.png`), then run the following code:

```python
python train.py --model TextImage3 --text_model clip --n_epochs 20 --dropout 0.5 --crop 0.25 --batch_size 32  --weight_decay 5e-5
```

#### Use the model
To use the model, please download the checkpoint via this [link](https://drive.google.com/file/d/1-Ohu8_FOFOfUxtV_Qho3g3hBHH-7_knR/view?usp=sharing), then put the file `textimage_rn50_clip.pt` into the folder `checkpoints` and run the following code:

```python
import torch
from torch.utils.data import DataLoader
from models.text_image import TextImage, TextImage2
from memes_dataset import MemesDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the checkpoint.
model = TextImage2(word_dim=512)
model.to(device)
checkpoint = torch.load('checkpoints/textimage_rn50_clip.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# - or - 
model.train()
```

#### Evaluate the model on the test set
```python
python evaluate_test.py 
```