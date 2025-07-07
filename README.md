# Image Captioning Using ResNet-LSTM with an Attention Mechanism 

An advanced image captioning model that combines ResNet-based CNN encoder, LSTM decoder, and attention mechanism to generate accurate image captions. Achieved a BLEU score of 0.68 using the Flickr8k dataset.

## 🚀 Project Overview 

The project implements an end-to-end image captioning system that:

- Uses pretrained ResNet-50 CNN as feature extractor (encoder).
- Applies LSTM-based RNN for sequence generation (decoder).
- Integrates attention mechanism to focus on relevant image regions.
- Processes and trains on the Flickr8k dataset with proper normalization and padding.

## 🔧 Key Features

- ResNet50 pretrained on ImageNet for feature extraction.
- LSTM decoder with attention mechanism to dynamically focus on image regions.
- Vocabulary creation with frequency threshold for word filtering.
- Sequence padding for uniform batch processing.
- Achieved BLEU score: 0.68 on Flickr8k dataset.

## 🏗 Technologies Used

- Python
- PyTorch
- torchvision
- spacy
- pandas
- numpy
- PIL
- TensorBoard (for logging)

## 🗃 Dataset

- Flickr8k dataset containing ~8000 images with 5 captions each.
- Image preprocessing: resized and normalized to 299x299 or 224x224.
- Captions are tokenized using the SpaCy tokenizer.
- Vocabulary built with frequency thresholding to filter rare words.
- Dataset loaded via custom PyTorch Dataset class and DataLoader.

## 🧑‍💻 Model Architecture

### 1. Encoder: ResNet50 CNN

- Loaded pretrained ResNet50.
- Removed final classification layers.
- Added AdaptiveAvgPool2D to produce (14x14) feature maps.
- Reduced feature channels via 1x1 convolution.
- Output reshaped to (Batch, 196, embed_size) for attention.

### 2. Attention Mechanism

- Computes attention weights using encoder output and decoder hidden state.
- Generates context vector as weighted sum of image features.
- Allows model to focus on different regions of the image for each generated word.

### 3. Decoder: LSTM with Attention

- Embeds input word using nn.Embedding layer.
- Concatenates context vector and word embedding as input to LSTMCell.
- Outputs word predictions at each time step using a linear layer.

### 4. Full Model Pipeline

- Images → EncoderCNN → Attention + DecoderRNN → Caption sequence.
- Trained with CrossEntropyLoss ignoring padding tokens.

## ⚙️ Training Details

- Optimizer: Adam
- Learning Rate: 3e-4
- Batch Size: 32
- Epochs: 100
- Device: GPU or CPU
- Loss: CrossEntropyLoss (ignoring <PAD> token)
- TensorBoard is used for monitoring loss.
  

  


