import os
import glob
import trimesh
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class PartDataset(Dataset):
    def __init__(self, root_dir, categories=None, max_parts=8, points_per_part=64, transform=None):
        """\
        Dataset for part-aware 3D shapes. В конструкторе сохраняем только пути к
        файлам и маски, а чтение мешей выполняем при обращении к элементу.
        """
        self.root_dir = root_dir
        self.categories = categories if categories is not None else []
        self.max_parts = max_parts
        self.points_per_part = points_per_part
        self.points_per_shape = max_parts * points_per_part
        self.transform = transform if transform is not None else T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.samples = []  # (img_path, [part_files], part_mask)
        categories_to_use = self.categories or [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for cat in categories_to_use:
            cat_dir = os.path.join(root_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            shape_dirs = [os.path.join(cat_dir, d) for d in os.listdir(cat_dir)
                          if os.path.isdir(os.path.join(cat_dir, d))]
            for shape_dir in shape_dirs:
                img_dir = os.path.join(shape_dir, "images")
                img_path = None
                if os.path.isdir(img_dir):
                    imgs = glob.glob(os.path.join(img_dir, "*.*"))
                    if imgs:
                        img_path = imgs[0]
                else:
                    cand = glob.glob(os.path.join(shape_dir, "*.png")) + glob.glob(os.path.join(shape_dir, "*.jpg"))
                    if cand:
                        img_path = cand[0]

                part_dir = os.path.join(shape_dir, "parts")
                if os.path.isdir(part_dir):
                    part_files = sorted(glob.glob(os.path.join(part_dir, "*.obj")))
                    if not part_files:
                        part_files = sorted(glob.glob(os.path.join(part_dir, "*.ply")))
                else:
                    part_files = sorted(glob.glob(os.path.join(shape_dir, "*.obj")))
                if not part_files:
                    continue
                num_parts = len(part_files)
                if num_parts > self.max_parts:
                    part_files = part_files[:self.max_parts]
                    num_parts = self.max_parts
                part_mask = [1] * num_parts + [0] * (self.max_parts - num_parts)
                self.samples.append((img_path, part_files, part_mask))
        print(f"Collected {len(self.samples)} shapes from {root_dir} (categories: {categories_to_use})")

    def __len__(self):
        return len(self.samples)

    def _load_mesh(self, path):
        try:
            mesh = trimesh.load(path, process=False)
            if isinstance(mesh, trimesh.Trimesh):
                return mesh
            if isinstance(mesh, trimesh.Scene):
                return trimesh.util.concatenate([
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in mesh.geometry.values()
                ])
        except Exception as e:
            print(f"Warning: failed to load mesh {path}: {e}")
        return None

    def __getitem__(self, idx):
        img_path, part_files, part_mask = self.samples[idx]
        if img_path is not None and os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        img_tensor = self.transform(img)

        part_meshes = []
        for pf in part_files:
            mesh = self._load_mesh(pf)
            if mesh is not None:
                part_meshes.append(mesh)
        num_parts = len(part_meshes)

        part_pts_list = []
        for mesh in part_meshes:
            pts = mesh.sample(self.points_per_part).astype(float)
            part_pts_list.append(torch.tensor(pts, dtype=torch.float))
        if num_parts < self.max_parts:
            for _ in range(self.max_parts - num_parts):
                part_pts_list.append(torch.zeros((self.points_per_part, 3), dtype=torch.float))
        part_points = torch.stack(part_pts_list[:self.max_parts]) if part_pts_list else torch.zeros((self.max_parts, self.points_per_part, 3), dtype=torch.float)
        part_mask_tensor = torch.tensor(part_mask, dtype=torch.bool)

        if part_meshes:
            try:
                combined_mesh = trimesh.util.concatenate(part_meshes)
            except Exception:
                combined_mesh = part_meshes[0]
                for m in part_meshes[1:]:
                    combined_mesh = combined_mesh.union(m) if hasattr(combined_mesh, 'union') else combined_mesh + m
            shape_pts = combined_mesh.sample(self.points_per_shape).astype(float)
            shape_points = torch.tensor(shape_pts, dtype=torch.float)
        else:
            shape_points = torch.zeros((self.points_per_shape, 3), dtype=torch.float)

        return img_tensor, part_points, shape_points, part_mask_tensor
