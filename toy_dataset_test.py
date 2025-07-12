from torch_geometric.datasets import GeometricShapes
from torch_geometric.transforms import SamplePoints
from dataloader import PartDataset
from training import chamfer_distance
from model import ShapeAsPointsPlusPlusModel
import torch, os, trimesh
from PIL import Image
from torch.utils.data import DataLoader

# 1. Загрузка и обработка данных
dataset = GeometricShapes(root='data/GeometricShapes')
dataset.transform = SamplePoints(num=64)

data = dataset[0]
print("Кол-во примеров:", len(dataset))
print("Пример точек:", data.pos.shape)

# 2. Сохраняем облако точек напрямую (.obj формат)
os.makedirs('toy/DataRoot/ToyShape/parts', exist_ok=True)
with open('toy/DataRoot/ToyShape/parts/part0.obj', 'w') as f:
    for p in data.pos.numpy():
        f.write(f"v {p[0]} {p[1]} {p[2]}\n")

# placeholder image
os.makedirs('toy/DataRoot/ToyShape/images', exist_ok=True)
img = Image.new('RGB', (224,224), color='gray')
img.save('toy/DataRoot/ToyShape/images/0.png')

# 3. Используем PartDataset
pd = PartDataset(root_dir='toy/DataRoot', max_parts=1, points_per_part=16, categories=['ToyShape'])
print("Датасет PartDataset:", len(pd))

loader = DataLoader(pd, batch_size=1)

# 4. Проверим модель и loss
for imgs, part_pts, shape_pts, part_mask in loader:
    model = ShapeAsPointsPlusPlusModel(points_per_shape=16)
    pred = model(imgs)
    loss = chamfer_distance(pred[0], shape_pts[0])
    print("Chamfer loss:", loss.item())
    break
