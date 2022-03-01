import torch
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
import seaborn as sns
from PIL import Image
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='flowers/test/12/image_04059.jpg')
    parser.add_argument('--checkpoints', type=str, default='checkpoint.pth')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--cat_to_name', type=str, default='cat_to_name.json')
    parser.add_argument('--gpu',  default=True)
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoints = torch.load(filepath)
    model = models.vgg19(pretrained=True)
    model.class_to_idx = checkpoints['class_to_idx']
    model.classifier = checkpoints['classifier']
    model.load_state_dict(checkpoints['state_dict'])
    return model

def process_image(image):
    resize = 256
    crop_size = 224
    (image_width, image_height) = image.size

    #resize and crop the image
    if image_height > image_width:
        image_height = int(max(image_height * resize / image_height, 1))
        image_width = int(resize)
    else:
        image_width = int(max(image_width * resize / image_height, 1))
        image_height = int(resize)

    image = image.resize((image_width, image_height))
    # crop image
    left = (image_width - crop_size) / 2
    top = (image_height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))

    # color channels
    image = np.array(image)
    image = image / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return image


def predict(image_path, model, top_k, gpu):
    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    image = Image.open(image_path)
    image = process_image(image)
    image = torch.from_numpy(image)
    image = image.unsqueeze_(0)
    image = image.float()

    with torch.no_grad():
        output = model.forward(image.cuda())

    p = F.softmax(output.data, dim=1)

    top_p = np.array(p.topk(top_k)[0][0])

    index_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [np.int(index_to_class[each]) for each in np.array(p.topk(top_k)[1][0])]

    return top_p, top_classes, device


def load_names(category_names_file):
    with open(category_names_file) as file:
        category_names = json.load(file)
    return category_names


def main():
    args = get_arguments()
    image_path = args.image_path
    checkpoint = args.checkpoints
    top_k = args.top_k
    cat_to_name = args.cat_to_name
    gpu = args.gpu

    model = load_checkpoint(checkpoint)

    top_p, classes, device = predict(image_path, model, top_k, gpu)

    #load the names from the file
    cat_to_name = load_names(cat_to_name)

    labels = [cat_to_name[str(index)] for index in classes]

    print(f"the prediction of the flower you have choosen: {image_path}")
    print(labels)
    print(top_p)

    for i in range(len(labels)):
        print("{} - {} with a probability of {}".format((i+1), labels[i], top_p[i]))


if __name__ == "__main__":
    main()