import os, glob
import trimesh
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class PartDataset(Dataset):
    def __init__(self, root_dir, categories=None, max_parts=8, points_per_part=64, transform=None):
        """
        Dataset for part-aware 3D shapes.
        - root_dir: path to dataset root containing category folders or shape folders.
        - categories: list of category names to include (e.g. ['Chair','Table']) or None for all.
        - max_parts: maximum number of parts to consider per shape (pad if shape has fewer parts).
        - points_per_part: number of surface sample points per part.
        - transform: image transform pipeline (resize, normalize, etc.).
        """
        self.root_dir = root_dir
        self.categories = categories if categories is not None else []
        self.max_parts = max_parts
        self.points_per_part = points_per_part
        self.points_per_shape = max_parts * points_per_part  # total points to represent whole shape
        # Image preprocessing: resize to 224x224 and normalize as ResNet expects
        self.transform = transform if transform is not None else T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.samples = []  # will hold (image_path, part_points_tensor, shape_points_tensor, part_mask_tensor)
        categories_to_use = self.categories
        if not categories_to_use:  # if no specific categories, use all in root
            categories_to_use = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for cat in categories_to_use:
            cat_dir = os.path.join(root_dir, cat)
            if not os.path.isdir(cat_dir):
                continue
            # Each shape in category folder
            shape_dirs = [os.path.join(cat_dir, d) for d in os.listdir(cat_dir) 
                          if os.path.isdir(os.path.join(cat_dir, d))]
            for shape_dir in shape_dirs:
                # Find image file (assume one image per shape for conditioning)
                img_dir = os.path.join(shape_dir, "images")
                img_path = None
                if os.path.isdir(img_dir):
                    # Take the first image file (or a specific view if multiple)
                    imgs = glob.glob(os.path.join(img_dir, "*.*"))
                    if imgs:
                        img_path = imgs[0]
                else:
                    # Or look for an image in the shape_dir directly
                    cand = glob.glob(os.path.join(shape_dir, "*.png")) + glob.glob(os.path.join(shape_dir, "*.jpg"))
                    if cand:
                        img_path = cand[0]
                # Find part mesh files
                part_dir = os.path.join(shape_dir, "parts")
                part_files = []
                if os.path.isdir(part_dir):
                    part_files = sorted(glob.glob(os.path.join(part_dir, "*.obj")))
                    if not part_files:  # maybe different ext
                        part_files = sorted(glob.glob(os.path.join(part_dir, "*.ply")))
                else:
                    # If no 'parts' subdir, assume all .obj in shape_dir are part files (or a single whole-object file)
                    part_files = sorted(glob.glob(os.path.join(shape_dir, "*.obj")))
                    # Exclude a file that might represent the whole object (if parts are separate).
                    # (Heuristic: if there's only one .obj, treat it as a single part. If multiple, assume each is a part.)
                if len(part_files) == 0:
                    continue  # no mesh found, skip
                # Load each part mesh and sample points
                part_meshes = []
                for pf in part_files:
                    try:
                        mesh = trimesh.load(pf, process=False)
                    except Exception as e:
                        # Skip if mesh fails to load
                        print(f"Warning: failed to load mesh {pf}: {e}")
                        continue
                    if isinstance(mesh, trimesh.Trimesh):
                        part_meshes.append(mesh)
                    elif isinstance(mesh, trimesh.Scene):
                        # If a Scene (multiple geometry), merge into one Trimesh
                        combined = trimesh.util.concatenate([trimesh.Trimesh(vertices=g.vertices, faces=g.faces) 
                                                             for g in mesh.geometry.values()])
                        part_meshes.append(combined)
                if len(part_meshes) == 0:
                    continue
                num_parts = len(part_meshes)
                if num_parts > self.max_parts:
                    # If shape has more parts than max_parts, we truncate (or could merge minor parts).
                    part_meshes = part_meshes[:self.max_parts]
                    num_parts = self.max_parts
                # Sample points_per_part on each part surface
                part_pts_list = []
                for mesh in part_meshes:
                    # Sample exactly points_per_part points uniformly on the surface
                    pts = mesh.sample(self.points_per_part)
                    pts = pts.astype(float)  # numpy array (P,3)
                    part_pts_list.append(torch.tensor(pts, dtype=torch.float))
                # If fewer parts than max, add dummy points (e.g. all zeros) for padding
                part_mask = [1]*num_parts + [0]*(self.max_parts - num_parts)
                if num_parts < self.max_parts:
                    # Pad part points with zeros for missing parts
                    for _ in range(self.max_parts - num_parts):
                        pad_pts = torch.zeros((self.points_per_part, 3), dtype=torch.float)
                        part_pts_list.append(pad_pts)
                # Stack part point clouds into a tensor [max_parts, points_per_part, 3]
                part_points = torch.stack(part_pts_list[:self.max_parts])  # shape (max_parts, P_per_part, 3)
                part_mask_tensor = torch.tensor(part_mask, dtype=torch.bool)
                # Sample points_per_shape from the entire object by merging all part meshes
                total_pts = self.points_per_shape
                try:
                    combined_mesh = trimesh.util.concatenate(part_meshes)
                except Exception:
                    # If concatenate fails (some meshes might not share material), merge manually
                    combined_mesh = part_meshes[0]
                    for m in part_meshes[1:]:
                        combined_mesh = combined_mesh.union(m) if hasattr(combined_mesh, 'union') else combined_mesh + m
                shape_pts = combined_mesh.sample(total_pts)
                shape_pts = shape_pts.astype(float)
                shape_points = torch.tensor(shape_pts, dtype=torch.float)
                # Save sample data (we will load image later in __getitem__)
                self.samples.append((img_path, part_points, shape_points, part_mask_tensor))
        print(f"Loaded {len(self.samples)} shapes from {root_dir} (categories: {categories_to_use})")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, part_points, shape_points, part_mask = self.samples[idx]
        # Load and transform image
        if img_path is not None and os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
        else:
            # If no image available, use a blank image or an array of zeros
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
        img_tensor = self.transform(img)  # normalized image tensor [3 x 224 x 224]
        # Return image, part-wise points, whole-shape points, and part mask
        return img_tensor, part_points, shape_points, part_mask
