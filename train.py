import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='flowers')
    parser.add_argument('--model_arch', type=str, default='vgg19', choices=['vgg19', 'densenet121'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--gpu',  default=True)
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth')
    return parser.parse_args()


def model_network(model_arch,gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_arch == "vgg19":
        model = models.vgg19(pretrained=True)
        features = 25088
    else:
        model = models.densenet121(pretrained=True)
        features = 1024

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(features,500),
                        nn.ReLU(),
                        nn.Dropout(0.25),
                        nn.Linear(500, 102),
                        nn.LogSoftmax(dim=1))

    model.to(device)
    return model,features,device


def train_model(epochs, trainloader, validloader, model, device, criterion, optimizer):
    steps = 0
    running_loss = 0
    print_every = 50

    for e in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)  

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)  
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                       f"Train loss: {running_loss/print_every:.3f}.. "
                       f"Valid loss: {test_loss/len(validloader):.3f}.. "
                       f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()


def save_checkpoint(filepath, model, image_datasets, epochs, optimizer, model_arch,output_size):
    model.class_to_idx = image_datasets[0].class_to_idx
    checkpoint = {
        'pretrained_model': model_arch,
        'output_size': output_size,
        'classifier': model.classifier,
        'epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, filepath)
   

def main():
    args = get_arguments()
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    image_datasets = [train_data, valid_data, test_data]

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    dataloaders = [trainloader, validloader, testloader]

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    model,features,device = model_network(args.model_arch,args.gpu)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0009)

    train_model(args.epochs, dataloaders[0], dataloaders[1], model, device, criterion, optimizer)

    filepath = args.save_dir

    output_size = 102
    save_checkpoint(filepath, model, image_datasets, args.epochs, optimizer, output_size, args.model_arch)

if __name__ == "__main__":
    main()