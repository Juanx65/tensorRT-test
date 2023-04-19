import os
import argparse

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

from models.mnist_resnet50 import resnet50


if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

current_directory = os.path.dirname(os.path.abspath(__file__))

# Hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
NUM_EPOCHS = 20

# Architecture
NUM_FEATURES = 28*28
NUM_CLASSES = 10

# Other
DEVICE = "cuda:0"
GRAYSCALE = True

device = torch.device(DEVICE)

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
        if(targets.size() == torch.Size([128])):
            features = features.to(device)
            targets = targets.to(device)

            logits, probas = model(features)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    #print("correctos: ", correct_pred ,'| total: ',num_examples )
    return correct_pred.float() / num_examples * 100

def main(args: argparse.Namespace) -> None:
    
    # Crear una instancia del modelo con la misma arquitectura
    model = resnet50(NUM_CLASSES)
    model.to(DEVICE)

    # Cargar los pesos del modelo
    model.load_state_dict(torch.load(args.weights))

    # Cambiar el modelo a eval() para usarlo en inferencia
    with torch.set_grad_enabled(False): # save memory during inference
        print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=args.device)))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', 
                        type=str,
                        default='weights/mnist.pth',
                        help='weights file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)