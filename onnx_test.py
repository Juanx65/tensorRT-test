import argparse
from io import BytesIO

import tensorflow as tf
import onnx
import tf2onnx
import torch

from models.mnist_resnet import resnet50
from models.mnist_resnet import resnet18
from models.mnist_resnet_keras import resnet18_mnist

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help='PyTorch resnet50 weights')
    parser.add_argument('--resnet',
                        type=str,
                        default='resnet18',
                        required=True,
                        help='resnet18 or resnet50')
    parser.add_argument('--library',
                        type=str,
                        default='tensorflow',
                        required=True,
                        help='pythorch or tensorflow')
    parser.add_argument('--opset',
                        type=int,
                        default=11,
                        help='ONNX opset version')
    parser.add_argument('--sim',
                        action='store_true',
                        help='simplify onnx model')
    parser.add_argument('--input-shape',
                        nargs='+',
                        type=int,
                        default=[128,1, 28, 28],
                        help='Model input shape only for api builder')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='Export ONNX device')
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def main(args):

    if(args.resnet == 'resnet18'):
        if(args.library == 'pytorch'):
            model = resnet18(10)# donde 10 es el numero de clases
        elif(args.library == 'tensorflow'):
            #print(args.input_shape[2:] + [args.input_shape[1]])
            model = resnet18_mnist(args.input_shape[2:] + [args.input_shape[1]],10)
        else:
            print("libreria no valida")
            return
    elif(args.resnet == 'resnet50'):
        model = resnet50(10)
    else:
        print('modelo no valido')
        return
    
    if(args.library == 'pytorch'):
        model.to(args.device)
        model.load_state_dict(torch.load(args.weights), strict=False)
        model.eval()
        fake_input = torch.randn(args.input_shape).to(args.device)
        for _ in range(2):
            model(fake_input)
        save_path = args.weights.replace('.pth', '.onnx')
        with BytesIO() as f:
            torch.onnx.export(
                model,
                fake_input,
                f,
                opset_version=args.opset,
                input_names=['images'],
                output_names=['logits', 'probas'])
            f.seek(0)
            onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)
    elif(args.library == 'tensorflow'):
        save_path = args.weights.replace('.h5', '.onnx')
        onnx_model, _ = tf2onnx.convert.from_keras(model=model, output_path=save_path)


    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')


if __name__ == '__main__':
    main(parse_args())