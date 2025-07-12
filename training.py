import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
import argparse

# Import the dataset and models defined above
from dataloader import PartDataset
from model import PartCrafterModel, ShapeAsPointsPlusPlusModel, PointCraftPlusPlusModel

def chamfer_distance(pc1, pc2):
    """
    Compute Chamfer Distance between two point clouds pc1 and pc2.
    pc1: Tensor [N1,3], pc2: Tensor [N2,3]
    Returns mean of squared distances.
    """
    # Pairwise distance matrix
    diff = pc1.unsqueeze(1) - pc2.unsqueeze(0)              # [N1, N2, 3]
    dist_sq = torch.sum(diff * diff, dim=2)                 # [N1, N2]
    # For each point in one cloud find nearest point in the other
    min_dist1, _ = torch.min(dist_sq, dim=1)                # [N1]
    min_dist2, _ = torch.min(dist_sq, dim=0)                # [N2]
    # Average of nearest distances (symmetrized)
    return torch.mean(min_dist1) + torch.mean(min_dist2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PartCrafter or baseline models")
    parser.add_argument('--model', type=str, choices=['partcrafter', 'shapeaspoints', 'pointcraft'], default='partcrafter', help="Which model to train")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset root directory")
    parser.add_argument('--category', type=str, default=None, help="Category to train on (e.g. Chair, Car, etc.) or all if None")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Training batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--max_parts', type=int, default=8, help="Max parts per shape (for part-based models)")
    parser.add_argument('--points_per_part', type=int, default=64, help="Points sampled per part for training")
    parser.add_argument('--num_workers', type=int, default=0, help="\u041f\u0440\u043e\u0446\u0435\u0441\u0441\u043e\u0432-\u0440\u0430\u0431\u043e\u0447\u0438\u0445 \u0434\u043b\u044f DataLoader")
    parser.add_argument('--pin_memory', action='store_true', help="\u0418\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u044c pinned memory")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--freeze_backbone', dest='freeze_backbone', action='store_true', help='Freeze image backbone (default)')
    parser.add_argument('--unfreeze_backbone', dest='freeze_backbone', action='store_false', help='Train backbone weights')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pretrained ResNet weights (default)')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='Do not load pretrained weights')
    parser.set_defaults(freeze_backbone=True, pretrained=True)
    args = parser.parse_args()

    # Load dataset and split into train/validation
    dataset = PartDataset(root_dir=args.data_dir,
                          categories=[args.category] if args.category else None,
                          max_parts=args.max_parts,
                          points_per_part=args.points_per_part)
    val_ratio = 0.1
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers,
                              pin_memory=args.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=args.num_workers,
                            pin_memory=args.pin_memory)
    # Instantiate model
    if args.model == 'partcrafter':
        model = PartCrafterModel(max_parts=args.max_parts,
                                 tokens_per_part=args.points_per_part,
                                 freeze_backbone=args.freeze_backbone,
                                 pretrained=args.pretrained)
    elif args.model == 'pointcraft':
        model = PointCraftPlusPlusModel(max_parts=args.max_parts,
                                        tokens_per_part=args.points_per_part,
                                        freeze_backbone=args.freeze_backbone,
                                        pretrained=args.pretrained)
    elif args.model == 'shapeaspoints':
        total_points = args.max_parts * args.points_per_part  # points_per_shape used in dataset
        model = ShapeAsPointsPlusPlusModel(points_per_shape=total_points,
                                           freeze_backbone=args.freeze_backbone,
                                           pretrained=args.pretrained)
    model = model.to(args.device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    # Training loop
    model.train()
    for epoch in range(1, args.epochs+1):
        total_loss = 0.0
        for batch in train_loader:
            imgs, part_pts, shape_pts, part_mask = batch
            imgs = imgs.to(args.device)                          # [B, 3, 224, 224]
            part_pts = part_pts.to(args.device)                  # [B, max_parts, pts_per_part, 3]
            shape_pts = shape_pts.to(args.device)                # [B, points_per_shape, 3]
            part_mask = part_mask.to(args.device)                # [B, max_parts]
            optimizer.zero_grad()
            # Forward pass
            outputs = model(imgs)
            # Compute loss (Chamfer distance)
            batch_loss = 0.0
            B = imgs.size(0)
            for i in range(B):
                if args.model == 'shapeaspoints':
                    pred_points = outputs[i]              # [points_per_shape, 3]
                    gt_points = shape_pts[i]              # [points_per_shape, 3]
                else:
                    # Part-based models
                    pred_parts = outputs[i]               # [max_parts, pts_per_part, 3]
                    gt_points = shape_pts[i]              # [points_per_shape, 3]
                    # Filter out non-existent part outputs using part_mask
                    valid_mask = part_mask[i]             # [max_parts]
                    # Concatenate points from all valid parts
                    pred_points = pred_parts[valid_mask].reshape(-1, 3)
                    # (GT shape points already include points from all real parts)
                # Chamfer distance between pred_points and gt_points
                loss_i = chamfer_distance(pred_points, gt_points)
                batch_loss += loss_i
            batch_loss = batch_loss / B
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * B
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs}, Training ChamferLoss = {avg_loss:.6f}")

        # Validation loop
        val_total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                imgs, part_pts, shape_pts, part_mask = batch
                imgs = imgs.to(args.device)
                part_pts = part_pts.to(args.device)
                shape_pts = shape_pts.to(args.device)
                part_mask = part_mask.to(args.device)
                outputs = model(imgs)
                batch_loss = 0.0
                B = imgs.size(0)
                for i in range(B):
                    if args.model == 'shapeaspoints':
                        pred_points = outputs[i]
                        gt_points = shape_pts[i]
                    else:
                        pred_parts = outputs[i]
                        gt_points = shape_pts[i]
                        valid_mask = part_mask[i]
                        pred_points = pred_parts[valid_mask].reshape(-1, 3)
                    loss_i = chamfer_distance(pred_points, gt_points)
                    batch_loss += loss_i
                batch_loss = batch_loss / B
                val_total_loss += batch_loss.item() * B
        val_avg_loss = val_total_loss / len(val_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs}, Validation ChamferLoss = {val_avg_loss:.6f}")
        model.train()
        # Save checkpoint periodically
        if epoch % 10 == 0:
            ckpt_path = f"{args.model}_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
    # Save final model
    torch.save(model.state_dict(), f"{args.model}_final.pth")
    print("Training complete, model saved.")
