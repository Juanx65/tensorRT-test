import argparse

from models.engine import EngineBuilder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',
                        type=str,
                        required=True,
                        help='Weights file')
    parser.add_argument('--input-shape',
                        nargs='+',
                        type=int,
                        default=[128, 1, 28, 28],
                        help='Model input shape, el primer valor es el batch_size, 128)]')
    parser.add_argument('--fp16',
                        action='store_true',
                        help='Build model with fp16 mode')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='TensorRT builder device')
    parser.add_argument('--seg',
                        action='store_true',
                        help='Build seg model by onnx')
    args = parser.parse_args()
    assert len(args.input_shape) == 4
    return args


def main(args):
    builder = EngineBuilder(args.weights, args.device)
    builder.seg = args.seg
    builder.build(fp16=args.fp16)

if __name__ == '__main__':
    args = parse_args()
    main(args)