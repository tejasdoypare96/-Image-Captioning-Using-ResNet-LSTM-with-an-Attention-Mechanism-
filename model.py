import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# Encoder: ResNet50
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))  # For attention
        self.conv = nn.Conv2d(2048, embed_size, kernel_size=1)
        self.bn = nn.BatchNorm2d(embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = self.adaptive_pool(features)  # (B, 2048, 14, 14)
        features = self.relu(self.bn(self.conv(features)))  # (B, embed_size, 14, 14)
        features = features.flatten(2).permute(0, 2, 1)  # (B, 196, embed_size)
        return features


# Attention module
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (B, num_pixels, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B, 1, att_dim)
        att = self.full_att(torch.tanh(att1 + att2)).squeeze(2)  # (B, num_pixels)
        alpha = torch.softmax(att, dim=1)  # attention weights
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha


# Decoder with attention
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, encoder_dim=256, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        caption_len = captions.size(1)
        vocab_size = self.fc.out_features

        embeddings = self.embedding(captions)
        h, c = torch.zeros(batch_size, hidden_size).to(captions.device), torch.zeros(batch_size, hidden_size).to(captions.device)
        outputs = torch.zeros(batch_size, caption_len, vocab_size).to(captions.device)

        for t in range(caption_len):
            context, _ = self.attention(encoder_out, h)
            lstm_input = torch.cat([embeddings[:, t], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(h))
            outputs[:, t] = output

        return outputs


# Full model wrapper
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, attention_dim)

    def forward(self, images, captions):
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        self.eval()
        result_caption = []
        with torch.no_grad():
            encoder_out = self.encoder(image.unsqueeze(0))  # (1, 196, embed)
            h, c = torch.zeros(1, hidden_size).to(image.device), torch.zeros(1, hidden_size).to(image.device)

            word = torch.tensor([vocabulary.stoi["<SOS>"]]).to(image.device)
            for _ in range(max_length):
                embedding = self.decoder.embedding(word)
                context, _ = self.decoder.attention(encoder_out, h)
                lstm_input = torch.cat([embedding.squeeze(1), context], dim=1)
                h, c = self.decoder.lstm(lstm_input, (h, c))
                output = self.decoder.fc(h)
                predicted = output.argmax(1)
                word = predicted.unsqueeze(1)
                token = vocabulary.itos[predicted.item()]
                if token == "<EOS>":
                    break
                result_caption.append(token)

        return result_caption
