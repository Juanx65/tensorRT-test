import argparse
from io import BytesIO

import onnx
import torch

from models.mnist_resnet50 import resnet50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        required=True,
                        help='PyTorch resnet50 weights')
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
    #assert len(args.input_shape) == 4
    return args


def main(args):
    model = resnet50(10)# donde 10 es el numero de clases
    model.to(args.device)
    model.load_state_dict(torch.load(args.weights))
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

    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, saved as {save_path}')


if __name__ == '__main__':
    main(parse_args())