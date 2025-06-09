import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN


def train():
    # Data Transformations
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    # Loading data
    train_loader, dataset = get_loader(
        root_folder = "flickr8k/images",
        annotations_file = "flickr8k/captions.txt",
        transform = transform,
        num_workers = 2,
    )

    # Device configuration and model parameter
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = False
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    #Initialization model, loss and optimizer
    model = CNNtoRNN(embed_size, hidden_szie, vocab_size, num_layers).to(device)
    critetion = nn.CrossEntropyLoss(ignore_index = dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters((), lr = learning_rate))


    #Configuring the CNN parameter
    '''Sets which parts of the CNN are trainable based on train_CNN.
    Only the final fully connected layer (fc.weight and fc.bias) is trainable unless train_CNN is True
    '''
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True


        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total = len(train_loader), leave = False):

            imgs = imgs.to(device)
            captions = captions.to(device)
            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            writer.add_scaler("Training loss", loss.item(), global_step = step)
            step += 1

            optimizer.zero_grad()
            loss.backwar(loss)
            optimizer.step()


if __name__ == "__main__":
    train