import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
import argparse
import random
import numpy as np
import logging
import sys
from tqdm.auto import tqdm
import os

# Import the dataset and models defined above
from dataloader import PartDataset
from model import PartCrafterModel, ShapeAsPointsPlusPlusModel, PointCraftPlusPlusModel

def chamfer_distance(pc1, pc2, mask1=None, mask2=None):
    """\
    Вычисляет Chamfer Distance между двумя облаками точек.

    Parameters
    ----------
    pc1 : Tensor[B, N1, 3]
        Первое облако точек.
    pc2 : Tensor[B, N2, 3]
        Второе облако точек.
    mask1 : Tensor[B, N1], optional
        Маска для pc1 (точки, где False, игнорируются).
    mask2 : Tensor[B, N2], optional
        Маска для pc2.

    Returns
    -------
    Tensor
        Среднее значение Chamfer Distance по батчу.
    """

    dist = torch.cdist(pc1, pc2)  # [B, N1, N2]

    if mask1 is not None:
        mask1_expand = mask1.unsqueeze(2).expand_as(dist)
        dist = dist.masked_fill(~mask1_expand, float('inf'))
    if mask2 is not None:
        mask2_expand = mask2.unsqueeze(1).expand_as(dist)
        dist = dist.masked_fill(~mask2_expand, float('inf'))

    min_dist1 = dist.min(dim=2).values  # [B, N1]
    min_dist2 = dist.min(dim=1).values  # [B, N2]

    if mask1 is not None:
        min_dist1 = min_dist1.masked_fill(~mask1, 0.0)
    if mask2 is not None:
        min_dist2 = min_dist2.masked_fill(~mask2, 0.0)

    if mask1 is not None:
        cd1 = (min_dist1.pow(2) * mask1).sum(dim=1) / mask1.sum(dim=1).clamp(min=1)
    else:
        cd1 = min_dist1.pow(2).mean(dim=1)

    if mask2 is not None:
        cd2 = (min_dist2.pow(2) * mask2).sum(dim=1) / mask2.sum(dim=1).clamp(min=1)
    else:
        cd2 = min_dist2.pow(2).mean(dim=1)

    return (cd1 + cd2).mean()

def f_score(pc_pred, pc_gt, threshold, mask_pred=None, mask_gt=None):
    """\
    Вычисляет F-Score между предсказанным и истинным облаками точек.

    pc_pred : Tensor[B, Np, 3]
        Предсказанные точки.
    pc_gt : Tensor[B, Ng, 3]
        Точки из ground truth.
    threshold : float
        Максимальное расстояние для учета точек.
    mask_pred : Tensor[B, Np], optional
        Маска для предсказанных точек.
    mask_gt : Tensor[B, Ng], optional
        Маска для GT точек.
    """

    dist = torch.cdist(pc_pred, pc_gt)

    if mask_pred is not None:
        m_pred = mask_pred.unsqueeze(2).expand_as(dist)
        dist = dist.masked_fill(~m_pred, float('inf'))
    if mask_gt is not None:
        m_gt = mask_gt.unsqueeze(1).expand_as(dist)
        dist = dist.masked_fill(~m_gt, float('inf'))

    min_pred = dist.min(dim=2).values
    min_gt = dist.min(dim=1).values

    precision_mask = (min_pred <= threshold).float()
    if mask_pred is not None:
        precision = (precision_mask * mask_pred).sum(dim=1) / mask_pred.sum(dim=1).clamp(min=1)
    else:
        precision = precision_mask.mean(dim=1)

    recall_mask = (min_gt <= threshold).float()
    if mask_gt is not None:
        recall = (recall_mask * mask_gt).sum(dim=1) / mask_gt.sum(dim=1).clamp(min=1)
    else:
        recall = recall_mask.mean(dim=1)

    f = 2 * precision * recall / (precision + recall + 1e-8)
    return f.mean()

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """Сохранить модель и оптимизаторы."""
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    if scheduler is not None:
        state['scheduler_state'] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scheduler=None, map_location=None):
    """Загрузить модель и оптимизаторы."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    return checkpoint.get('epoch', 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PartCrafter or baseline models")
    parser.add_argument('--model', type=str, choices=['partcrafter', 'shapeaspoints', 'pointcraft'], default='partcrafter', help="Which model to train")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset root directory")
    parser.add_argument('--category', type=str, default=None, help="Category to train on (e.g. Chair, Car, etc.) or all if None")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="Training batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--lr_schedule', choices=['step', 'cosine'], default=None,
                        help='Тип LR scheduler: step или cosine')
    parser.add_argument('--seed', type=int, default=None,
                        help='Фиксировать начальное значение генератора случайных чисел')
    parser.add_argument('--max_parts', type=int, default=8, help="Max parts per shape (for part-based models)")
    parser.add_argument('--points_per_part', type=int, default=64, help="Points sampled per part for training")
    parser.add_argument('--fscore_thresh', type=float, default=0.01,
                        help="\u041f\u043e\u0440\u043e\u0433 \u0434\u043b\u044f \u0432\u044b\u0447\u0438\u0441\u043b\u0435\u043d\u0438\u044f F-Score")
    parser.add_argument('--cache_dir', type=str, default=None,
                        help="\u0414\u0438\u0440\u0435\u043a\u0442\u043e\u0440\u0438\u044f \u0434\u043b\u044f \u043a\u044d\u0448\u0430 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u043d\u043d\u044b\u0445 \u043c\u0435\u0448\u0435\u0439")
    parser.add_argument('--num_workers', type=int, default=0, help="\u041f\u0440\u043e\u0446\u0435\u0441\u0441\u043e\u0432-\u0440\u0430\u0431\u043e\u0447\u0438\u0445 \u0434\u043b\u044f DataLoader")
    parser.add_argument('--pin_memory', action='store_true', help="\u0418\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u0442\u044c pinned memory")
    parser.add_argument('--log_file', type=str, default=None, help='\u0424\u0430\u0439\u043b \u0434\u043b\u044f \u0441\u043e\u0445\u0440\u0430\u043d\u0435\u043d\u0438\u044f \u043b\u043e\u0433\u043e\u0432')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--freeze_backbone', dest='freeze_backbone', action='store_true', help='Freeze image backbone (default)')
    parser.add_argument('--unfreeze_backbone', dest='freeze_backbone', action='store_false', help='Train backbone weights')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Use pretrained ResNet weights (default)')
    parser.add_argument('--no_pretrained', dest='pretrained', action='store_false', help='Do not load pretrained weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Путь к файлу чекпоинта для продолжения обучения')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Сохранять чекпоинт каждые N эпох')
    parser.set_defaults(freeze_backbone=True, pretrained=True)
    args = parser.parse_args()

    log_handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_file:
        log_handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=log_handlers)
    logger = logging.getLogger(__name__)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load dataset and split into train/validation
    dataset = PartDataset(root_dir=args.data_dir,
                          categories=[args.category] if args.category else None,
                          max_parts=args.max_parts,
                          points_per_part=args.points_per_part,
                          cache_dir=args.cache_dir)
    val_ratio = 0.1
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    if train_size == 0:
        train_size = 1
        val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_drop_last = len(train_dataset) >= args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=train_drop_last,
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
    scheduler = None
    if args.lr_schedule == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    if args.resume is not None and os.path.isfile(args.resume):
        last_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=args.device)
        start_epoch = last_epoch + 1
        print(f"Возобновляем обучение с эпохи {start_epoch}")

    # Training loop
    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}"):
            imgs, part_pts, shape_pts, part_mask = batch
            imgs = imgs.to(args.device)                          # [B, 3, 224, 224]
            part_pts = part_pts.to(args.device)                  # [B, max_parts, pts_per_part, 3]
            shape_pts = shape_pts.to(args.device)                # [B, points_per_shape, 3]
            part_mask = part_mask.to(args.device)                # [B, max_parts]
            optimizer.zero_grad()
            # Forward pass
            outputs = model(imgs)
            # Compute loss (Chamfer distance) для всего батча
            if args.model == 'shapeaspoints':
                pred_points = outputs                             # [B, N, 3]
                pred_mask = None
            else:
                pred_parts = outputs                              # [B, max_parts, pts_per_part, 3]
                B = pred_parts.size(0)
                pred_points = pred_parts.view(B, -1, 3)
                # Маска валидных токенов
                pred_mask = part_mask.unsqueeze(-1).repeat(1, 1, args.points_per_part)
                pred_mask = pred_mask.view(B, -1)
            batch_loss = chamfer_distance(pred_points, shape_pts, mask1=pred_mask)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * imgs.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch}/{args.epochs}, Training ChamferLoss = {avg_loss:.6f}")

        # Validation loop
        val_total_loss = 0.0
        val_total_fscore = 0.0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val {epoch}/{args.epochs}"):
                imgs, part_pts, shape_pts, part_mask = batch
                imgs = imgs.to(args.device)
                part_pts = part_pts.to(args.device)
                shape_pts = shape_pts.to(args.device)
                part_mask = part_mask.to(args.device)
                outputs = model(imgs)
                if args.model == 'shapeaspoints':
                    pred_points = outputs
                    pred_mask = None
                else:
                    pred_parts = outputs
                    B = pred_parts.size(0)
                    pred_points = pred_parts.view(B, -1, 3)
                    pred_mask = part_mask.unsqueeze(-1).repeat(1, 1, args.points_per_part)
                    pred_mask = pred_mask.view(B, -1)
                batch_loss = chamfer_distance(pred_points, shape_pts, mask1=pred_mask)
                val_total_loss += batch_loss.item() * imgs.size(0)
                batch_f = f_score(pred_points, shape_pts, args.fscore_thresh, mask_pred=pred_mask)
                val_total_fscore += batch_f.item() * imgs.size(0)
        val_avg_loss = val_total_loss / len(val_loader.dataset)
        val_avg_f = val_total_fscore / len(val_loader.dataset)
        logger.info(f"Epoch {epoch}/{args.epochs}, Validation ChamferLoss = {val_avg_loss:.6f}, F-Score = {val_avg_f:.4f}")
        if scheduler is not None:
            scheduler.step()
        model.train()
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = f"{args.model}_epoch{epoch}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path)
            logger.info(f"Сохранён чекпоинт: {ckpt_path}")

    # Save final model
    ckpt_path = f"{args.model}_final.pth"
    save_checkpoint(model, optimizer, scheduler, args.epochs, ckpt_path)
    logger.info("Обучение завершено, финальная модель сохранена.")

