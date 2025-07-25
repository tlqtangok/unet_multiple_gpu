# python export_onnx.py --checkpoint_path ./checkpoints/checkpoint_epoch1.pth --batch_size 32 --height 160 --width 160

import torch
import torch.onnx
import argparse
from unet import UNet

def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.copy()
        state_dict.pop('mask_values', None)
        for key in ['epoch', 'optimizer', 'scheduler', 'loss']:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint

    return state_dict

def detect_model_config(state_dict):
    n_channels = 3
    n_classes = 1

    for name, param in state_dict.items():
        if 'inc.double_conv.0.weight' in name:
            n_channels = param.shape[1]
        elif 'outc.conv.weight' in name:
            n_classes = param.shape[0]

    return n_channels, n_classes

def create_model(n_channels, n_classes):
    return UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)

def load_weights(model, state_dict):
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Weights loaded successfully")
    except RuntimeError:
        model.load_state_dict(state_dict, strict=False)
        print("Weights loaded with strict=False")

def export_onnx(model, output_path):
    model.eval()

    dummy_input = torch.randn(1, 3, 160, 160)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=10,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            verbose=False
        )

def convert_model(checkpoint_path, output_path):
    state_dict = load_checkpoint(checkpoint_path)
    n_channels, n_classes = detect_model_config(state_dict)

    print(f"Model config: channels={n_channels}, classes={n_classes}")

    model = create_model(n_channels, n_classes)
    load_weights(model, state_dict)

    export_onnx(model, output_path)

    print(f"Model exported to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    convert_model(args.checkpoint, args.output)

if __name__ == '__main__':
    main()

