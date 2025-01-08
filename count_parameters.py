import torch
import argparse
from model import VisionTransformer, VisionTransformer_pytorch

def update_args(args):
    args.n_patches = (args.image_size // args.patch_size) ** 2
    args.is_cuda = torch.cuda.is_available()
    return args

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count parameters of Vision Transformer (ViT) model')

    # Training Arguments
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes in the dataset')

    # Data arguments
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashionmnist', 'svhn', 'cifar10', 'cifar100'], help='dataset to use')
    parser.add_argument("--image_size", type=int, default=28, help='image size')
    parser.add_argument("--patch_size", type=int, default=4, help='patch Size')
    parser.add_argument("--n_channels", type=int, default=1, help='number of channels')

    # ViT Arguments
    parser.add_argument("--use_torch_transformer_layers", type=bool, default=False, help="Use PyTorch Transformer Encoder layers instead of using scratch implementation.")
    parser.add_argument("--embed_dim", type=int, default=64, help='dimensionality of the latent space')
    parser.add_argument("--n_attention_heads", type=int, default=4, help='number of heads to use in Multi-head attention')
    parser.add_argument("--forward_mul", type=int, default=2, help='forward multiplier')
    parser.add_argument("--n_layers", type=int, default=6, help='number of encoder layers')
    parser.add_argument("--dropout", type=float, default=0.1, help='dropout value')

    args = parser.parse_args()
    args = update_args(args)

    # Create model
    if args.use_torch_transformer_layers:
        model = VisionTransformer_pytorch(n_channels=args.n_channels, embed_dim=args.embed_dim,
                                        n_layers=args.n_layers, n_attention_heads=args.n_attention_heads,
                                        forward_mul=args.forward_mul, image_size=args.image_size,
                                        patch_size=args.patch_size, n_classes=args.n_classes,
                                        dropout=args.dropout)
    else:
        model = VisionTransformer(n_channels=args.n_channels, embed_dim=args.embed_dim,
                                n_layers=args.n_layers, n_attention_heads=args.n_attention_heads,
                                forward_mul=args.forward_mul, image_size=args.image_size,
                                patch_size=args.patch_size, n_classes=args.n_classes,
                                dropout=args.dropout)

    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Configuration:")
    print(f"Dataset: {args.dataset}")
    print(f"Image size: {args.image_size}")
    print(f"Patch size: {args.patch_size}")
    print(f"Number of patches: {args.n_patches}")
    print(f"Embedding dimension: {args.embed_dim}")
    print(f"Number of layers: {args.n_layers}")
    print(f"Number of attention heads: {args.n_attention_heads}")
    print(f"Using PyTorch transformer layers: {args.use_torch_transformer_layers}")
    print(f"\nNumber of trainable parameters: {n_parameters:,}")
