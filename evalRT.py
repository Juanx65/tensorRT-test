from models import engine
import argparse
import time

import torch

import os

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

current_directory = os.path.dirname(os.path.abspath(__file__))

BATCH_SIZE = 128

test_dataset = datasets.MNIST(root='data', 
                              train=False, 
                              transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=BATCH_SIZE, 
                         shuffle=False)
def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        if( predicted_labels.size() ==targets.size() ):
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

def main(args: argparse.Namespace) -> None:

    #start_time = time.time()

    device = torch.device(args.device)
    engine_path = os.path.join(current_directory,args.engine)
    Engine = engine.TRTModule(engine_path, device)
    H, W = Engine.inp_info[0].shape[-2:]

    # set desired output names order
    Engine.set_desired(['logits', 'probas'])

    with torch.set_grad_enabled(False): # save memory during inference
        compute_accuracy(Engine, test_loader, device='cuda:0')
        #print('Test accuracy: %.2f%%' % (compute_accuracy(Engine, test_loader, device='cuda:0')))

    #end_time = time.time()
    #print("Elapsed time: {:.2f} seconds".format(end_time - start_time))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', 
                        type=str,
                        default='weights/mnist.engine',
                        help='Engine file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT infer device')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)