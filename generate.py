import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import PartCrafterModel, ShapeAsPointsPlusPlusModel, PointCraftPlusPlusModel

# Color palette for parts (RGB tuples)
PART_COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (255,128,0), (128,0,255)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 3D shape from a single image using a trained model")
    parser.add_argument('--model', type=str, choices=['partcrafter','shapeaspoints','pointcraft'], required=True, help="Model type")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to trained model weights (.pth file)")
    parser.add_argument('--image', type=str, required=True, help="Path to input image file")
    parser.add_argument('--max_parts', type=int, default=8, help="Max parts (must match training)")
    parser.add_argument('--points_per_part', type=int, default=64, help="Points per part (must match training)")
    parser.add_argument('--output', type=str, default="output.ply", help="Output PLY file name for the generated point cloud")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--freeze_backbone', dest='freeze_backbone', action='store_true', help='Freeze image backbone (default)')
    parser.add_argument('--unfreeze_backbone', dest='freeze_backbone', action='store_false', help='Load backbone with gradients enabled')
    parser.set_defaults(freeze_backbone=True)
    args = parser.parse_args()

    # Initialize the model architecture (same parameters as during training)
    if args.model == 'partcrafter':
        model = PartCrafterModel(max_parts=args.max_parts, tokens_per_part=args.points_per_part, freeze_backbone=args.freeze_backbone)
    elif args.model == 'pointcraft':
        model = PointCraftPlusPlusModel(max_parts=args.max_parts, tokens_per_part=args.points_per_part, freeze_backbone=args.freeze_backbone)
    elif args.model == 'shapeaspoints':
        total_points = args.max_parts * args.points_per_part
        model = ShapeAsPointsPlusPlusModel(points_per_shape=total_points, freeze_backbone=args.freeze_backbone)

    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model = model.to(args.device)
    model.eval()

    # Load and preprocess the input image
    img = Image.open(args.image).convert('RGB')
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(args.device)  # shape [1,3,224,224]

    # Generate shape
    with torch.no_grad():
        output = model(img_tensor)
    output = output.cpu().detach()
    # Prepare points and colors for saving
    if args.model == 'shapeaspoints':
        # output is [1, N, 3]
        points = output[0].numpy()  # (N,3)
        colors = [(255,255,255)] * points.shape[0]  # all points white
    else:
        # output is [1, max_parts, pts_per_part, 3]
        output = output[0]  # [max_parts, pts_per_part, 3]
        points_list = []
        colors = []
        for part_idx in range(args.max_parts):
            part_points = output[part_idx].numpy()
            # If the model used padding, it might output trivial points for non-existent parts.
            # Here we simply include all parts. Optionally, you could skip parts that are empty.
            color = PART_COLORS[part_idx % len(PART_COLORS)]
            for pt in part_points:
                points_list.append(pt)
                colors.append(color)
        points = points_list if isinstance(points_list, list) else points_list.numpy()
        points = np.array(points)  # ensure numpy array
    # Write to PLY file
    num_points = points.shape[0]
    with open(args.output, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x,y,z), (r,g,b) in zip(points, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r:d} {g:d} {b:d}\n")
    print(f"Saved generated shape to {args.output}")
