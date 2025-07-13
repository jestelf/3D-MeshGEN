import os
import argparse
import urllib.request
import zipfile
import glob
import shutil

URL = "https://modelnet.cs.princeton.edu/ModelNet40.zip"


def download(url, dst):
    if os.path.exists(dst):
        print(f"Файл {dst} уже загружен")
        return
    print(f"Скачиваем {url} ...")
    urllib.request.urlretrieve(url, dst)
    print("Готово")


def extract(zip_path, out_dir):
    print(f"Распаковываем {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    print("Готово")


def reorganize(src_root, dst_root):
    print(f"Подготовка датасета в {dst_root} ...")
    for category in os.listdir(src_root):
        cat_dir = os.path.join(src_root, category)
        if not os.path.isdir(cat_dir):
            continue
        for split in ["train", "test"]:
            split_dir = os.path.join(cat_dir, split)
            if not os.path.isdir(split_dir):
                continue
            off_files = glob.glob(os.path.join(split_dir, "*.off"))
            for off_file in off_files:
                shape_name = os.path.splitext(os.path.basename(off_file))[0]
                dst_shape = os.path.join(dst_root, category, shape_name, "parts")
                os.makedirs(dst_shape, exist_ok=True)
                dst_file = os.path.join(dst_shape, "part0.off")
                shutil.copy2(off_file, dst_file)
    print("Датасет готов")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузка и подготовка ModelNet40")
    parser.add_argument("--out_dir", type=str, default="data/ModelNet40", help="Куда поместить итоговый датасет")
    parser.add_argument("--zip_path", type=str, default="ModelNet40.zip", help="Путь для скачанного архива")
    args = parser.parse_args()

    download(URL, args.zip_path)
    extract(args.zip_path, os.path.dirname(args.out_dir))
    src_root = os.path.join(os.path.dirname(args.out_dir), "ModelNet40")
    reorganize(src_root, args.out_dir)
    print("Готово. Используйте", args.out_dir, "как --data_dir при обучении")
