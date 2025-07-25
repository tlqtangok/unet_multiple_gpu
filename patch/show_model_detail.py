# python show_model_detail.py  --model ./checkpoints/checkpoint_epoch1.onnx --detail
# python show_model_detail.py  --model ./checkpoints/checkpoint_epoch1.pth --detail

import torch
import onnx
import onnx.helper
import argparse
import os

def detect_model_type(model_path):
    """
    Detect model format based on file extension
    """
    ext = os.path.splitext(model_path)[1].lower()
    if ext == '.onnx':
        return 'onnx'
    elif ext == '.pth' or ext == '.pt':
        return 'pth'
    else:
        return 'unknown'

def load_pth_model(model_path):
    """
    Load PyTorch model from checkpoint
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        return checkpoint
    except Exception as e:
        print(f"Error loading PTH model: {e}")
        return None

def load_onnx_model(model_path):
    """
    Load ONNX model from file
    """
    try:
        model = onnx.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return None

def analyze_pth_structure(checkpoint):
    """
    Analyze PyTorch checkpoint structure
    """
    print("=" * 60)
    print("PTH MODEL STRUCTURE")
    print("=" * 60)
    
    if isinstance(checkpoint, dict):
        print("Checkpoint type: Dictionary")
        print(f"Keys: {list(checkpoint.keys())}")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        print("Checkpoint type: Direct model")
        if hasattr(checkpoint, 'state_dict'):
            state_dict = checkpoint.state_dict()
        else:
            state_dict = checkpoint
    
    return state_dict

def count_pth_parameters(state_dict):
    """
    Count parameters in PyTorch state dict
    """
    total_params = 0
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            total_params += param.numel()
    return total_params

def analyze_pth_layers(state_dict):
    """
    Analyze PyTorch model layers
    """
    print("\n" + "=" * 60)
    print("PTH LAYER INFORMATION")
    print("=" * 60)
    
    layer_types = {}
    layer_info = []
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            layer_info.append((name, param.shape, param.numel()))
            
            if 'conv' in name.lower():
                layer_types['Conv'] = layer_types.get('Conv', 0) + 1
            elif 'bn' in name.lower() or 'batch' in name.lower():
                layer_types['BatchNorm'] = layer_types.get('BatchNorm', 0) + 1
            elif 'fc' in name.lower() or 'linear' in name.lower():
                layer_types['Linear'] = layer_types.get('Linear', 0) + 1
            elif 'bias' not in name.lower():
                layer_types['Other'] = layer_types.get('Other', 0) + 1
    
    print(f"Total Parameters: {count_pth_parameters(state_dict):,}")
    print(f"Total Layers: {len(layer_info)}")
    
    print("\nLayer Type Distribution:")
    for layer_type, count in layer_types.items():
        print(f"  {layer_type}: {count}")

def print_pth_details(state_dict, show_details):
    """
    Print detailed PTH layer information
    """
    if not show_details:
        return
    
    print("\n" + "=" * 60)
    print("DETAILED PTH LAYERS")
    print("=" * 60)
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            print(f"Layer: {name}")
            print(f"  Shape: {list(param.shape)}")
            print(f"  Parameters: {param.numel():,}")
            print(f"  Data Type: {param.dtype}")
            print()

def detect_pth_model_config(state_dict):
    """
    Detect model configuration from state dict
    """
    n_channels = 3
    n_classes = 1
    
    for name, param in state_dict.items():
        if 'inc.double_conv.0.weight' in name:
            n_channels = param.shape[1]
        elif 'outc.conv.weight' in name:
            n_classes = param.shape[0]
    
    return n_channels, n_classes

def print_pth_config(state_dict):
    """
    Print PTH model configuration
    """
    print("\n" + "=" * 60)
    print("PTH MODEL CONFIGURATION")
    print("=" * 60)
    
    n_channels, n_classes = detect_pth_model_config(state_dict)
    print(f"Input Channels: {n_channels}")
    print(f"Output Classes: {n_classes}")

def analyze_onnx_inputs(graph):
    """
    Analyze ONNX model inputs
    """
    print("\n" + "=" * 60)
    print("ONNX MODEL INPUTS")
    print("=" * 60)
    
    for i, input_tensor in enumerate(graph.input):
        print(f"Input {i+1}:")
        print(f"  Name: {input_tensor.name}")
        
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(str(dim.dim_value))
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        
        print(f"  Shape: [{', '.join(shape)}]")

def analyze_onnx_outputs(graph):
    """
    Analyze ONNX model outputs
    """
    print("\n" + "=" * 60)
    print("ONNX MODEL OUTPUTS")
    print("=" * 60)
    
    for i, output_tensor in enumerate(graph.output):
        print(f"Output {i+1}:")
        print(f"  Name: {output_tensor.name}")
        
        shape = []
        for dim in output_tensor.type.tensor_type.shape.dim:
            if dim.dim_value:
                shape.append(str(dim.dim_value))
            elif dim.dim_param:
                shape.append(dim.dim_param)
            else:
                shape.append("?")
        
        print(f"  Shape: [{', '.join(shape)}]")

def count_onnx_parameters(graph):
    """
    Count ONNX model parameters
    """
    total_params = 0
    
    for initializer in graph.initializer:
        param_count = 1
        for dim in initializer.dims:
            param_count *= dim
        total_params += param_count
    
    return total_params

def analyze_onnx_layers(graph):
    """
    Analyze ONNX model layers
    """
    print("\n" + "=" * 60)
    print("ONNX MODEL LAYERS")
    print("=" * 60)
    
    layer_counts = {}
    total_layers = len(graph.node)
    
    for node in graph.node:
        op_type = node.op_type
        layer_counts[op_type] = layer_counts.get(op_type, 0) + 1
    
    print(f"Total Layers: {total_layers}")
    print(f"Total Parameters: {count_onnx_parameters(graph):,}")
    
    print("\nLayer Distribution:")
    for op_type, count in sorted(layer_counts.items()):
        print(f"  {op_type}: {count}")

def print_onnx_details(graph, show_details):
    """
    Print detailed ONNX layer information
    """
    if not show_details:
        return
    
    print("\n" + "=" * 60)
    print("DETAILED ONNX LAYERS")
    print("=" * 60)
    
    for i, node in enumerate(graph.node):
        print(f"Layer {i+1}: {node.op_type}")
        print(f"  Name: {node.name}")
        print(f"  Inputs: {list(node.input)}")
        print(f"  Outputs: {list(node.output)}")
        
        if node.attribute:
            print("  Attributes:")
            for attr in node.attribute:
                print(f"    {attr.name}: {onnx.helper.get_attribute_value(attr)}")
        print()

def analyze_pth_model(model_path, show_details):
    """
    Analyze PyTorch PTH model
    """
    checkpoint = load_pth_model(model_path)
    if checkpoint is None:
        return False
    
    state_dict = analyze_pth_structure(checkpoint)
    analyze_pth_layers(state_dict)
    print_pth_config(state_dict)
    print_pth_details(state_dict, show_details)
    
    return True

def analyze_onnx_model(model_path, show_details):
    """
    Analyze ONNX model
    """
    model = load_onnx_model(model_path)
    if model is None:
        return False
    
    graph = model.graph
    analyze_onnx_inputs(graph)
    analyze_onnx_outputs(graph)
    analyze_onnx_layers(graph)
    print_onnx_details(graph, show_details)
    
    return True

def create_parser():
    """
    Create command line argument parser
    """
    parser = argparse.ArgumentParser(description='Analyze ONNX and PTH models')
    parser.add_argument('--model', '-m', required=True,
                       help='Path to model file (.onnx or .pth)')
    parser.add_argument('--details', '-d', action='store_true',
                       help='Show detailed layer information')
    
    return parser

def validate_model_path(model_path):
    """
    Validate model file path
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_type = detect_model_type(model_path)
    if model_type == 'unknown':
        raise ValueError(f"Unsupported model format: {model_path}")
    
    return model_type

def analyze_model(model_path, show_details):
    """
    Main model analysis function
    """
    model_type = validate_model_path(model_path)
    
    print(f"Analyzing {model_type.upper()} model: {model_path}")
    
    if model_type == 'pth':
        return analyze_pth_model(model_path, show_details)
    elif model_type == 'onnx':
        return analyze_onnx_model(model_path, show_details)
    
    return False

def main():
    """
    Main entry point
    """
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        success = analyze_model(args.model, args.details)
        return 0 if success else 1
    except Exception as e:
        print(f"Analysis failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())


