import torch
import hls4ml
import tensorflow as tf

from models.mnist_resnet import resnet18

DEVICE = 'cuda:0'
NUM_CLASSES = 10

## Carga el modelo PyTorch
model_torch = resnet18(NUM_CLASSES)
model_torch.to(DEVICE)
state_dict = torch.load("/home/juan/Documents/tensorRT-test/weights/mnist18.pth")
model_torch.load_state_dict(state_dict)
model_torch.eval()

hls_model = hls4ml.converters.convert_from_pytorch_model(model=model_torch, input_shape=(1,28,28))
