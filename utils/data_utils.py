import os
import gzip
import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)
class MyDatasetPro(Dataset):
    def __init__(self, img_dir, sids, positions, targets):
        """
        img_dir   : 이미지 npz 파일이 있는 폴더
        sids      : (N,) 문자열 배열
        positions : (N,2) float32 numpy 배열
        targets   : (N,) float32 numpy 배열
        """
        self.img_dir = img_dir
        self.sids = np.asarray(sids)
        self.positions = np.asarray(positions, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, index):
        # -----------------------------
        # 1) sid → 파일명 변환
        # -----------------------------
        sid = str(self.sids[index])
        sid_prefix = sid.split("_19ch")[0]     # <-- 사용자가 알려준 규칙
        file_name = f"{sid_prefix}_19ch.npz"
        img_path = os.path.join(self.img_dir, file_name)

        # -----------------------------
        # 2) npz 파일 로드
        # -----------------------------

        npz = np.load(img_path)
        img_np = npz["combined_data"]
        img = torch.tensor(img_np, dtype=torch.float32)

        # -----------------------------
        # 3) 위치 / GVI label
        # -----------------------------
        pos = torch.tensor(self.positions[index], dtype=torch.float32)
        veg = torch.tensor([self.targets[index]], dtype=torch.float32)

        return img, pos, veg

# def get_loader(args):
#     img_dir = args.img_dir
#     data = pd.read_csv(args.dataset_csv)

#     # Split dataset into training (type=0) and eval (type=1)
#     train_data = data[data['type'] == 0]
#     test_data = data[data['type'] == 1]

#     # Select relevant columns: sid, longitude, latitude, gvi
#     cols = ['sid', 'longitude', 'latitude', 'gvi']
#     train_array = train_data[cols].values
#     test_array = test_data[cols].values

#     # Normalize coordinates to [0, 1] range
#     lon_min, lon_max = 102, 130.5
#     lat_min, lat_max = 22, 46.5

#     train_array[:, 1] = (train_array[:, 1] - lon_min) / (lon_max - lon_min)
#     test_array[:, 1] = (test_array[:, 1] - lon_min) / (lon_max - lon_min)

#     train_array[:, 2] = (train_array[:, 2] - lat_min) / (lat_max - lat_min)
#     test_array[:, 2] = (test_array[:, 2] - lat_min) / (lat_max - lat_min)

#     # Create dataset instances
#     train_dataset = MyDatasetPro(img_dir, train_array)
#     test_dataset = MyDatasetPro(img_dir, test_array)

#     # Create data loaders
#     train_loader = DataLoader(
#         train_dataset,
#         sampler=RandomSampler(train_dataset),
#         batch_size=args.train_batch_size,
#         num_workers=5,
#         pin_memory=True
#     )

#     eval_loader = DataLoader(
#         test_dataset,
#         sampler=SequentialSampler(test_dataset),
#         batch_size=args.eval_batch_size,
#         num_workers=5,
#         pin_memory=True
#     )

#     return train_loader, eval_loader


def get_loader(args):
    img_dir = args.img_dir
    data = pd.read_csv(args.dataset_csv)

    # 어떤 컬럼을 타깃으로 쓸지: 기본은 'gvi'
    target_col = getattr(args, "target_col", "gvi")

    if target_col not in data.columns:
        raise KeyError(
            f"target_col '{target_col}' not found in CSV. "
            f"Available columns: {list(data.columns)}"
        )

    # train/test split: type=0(train), type=1(test) 사용
    train_data = data[data["split"]==0].copy()
    test_data = data[data["split"] == 1].copy()

    # 좌표 정규화 범위 (원래 코드 그대로 사용)
    lon_min, lon_max = 102, 130.5
    lat_min, lat_max = 22, 46.5

    def prepare_split(df):
        # sid는 문자열
        sids = df["sid"].astype(str).values

        # float32 좌표
        lon = df["longitude"].astype("float32").values
        lat = df["latitude"].astype("float32").values

        # 정규화
        lon_norm = (lon - lon_min) / (lon_max - lon_min)
        lat_norm = (lat - lat_min) / (lat_max - lat_min)
        positions = np.stack([lon_norm, lat_norm], axis=1).astype("float32")

        # 타깃 (gvi 또는 extended_gvi)
        targets = df[target_col].astype("float32").values

        return sids, positions, targets

    train_sids, train_pos, train_targets = prepare_split(train_data)
    test_sids, test_pos, test_targets = prepare_split(test_data)

    # Dataset 생성 (새로운 MyDatasetPro 시그니처 사용)
    train_dataset = MyDatasetPro(img_dir, train_sids, train_pos, train_targets)
    test_dataset = MyDatasetPro(img_dir, test_sids, test_pos, test_targets)

    # num_workers 는 Colab이면 2 정도로 줄이는 게 안전
    num_workers = getattr(args, "num_workers", 2)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, eval_loader

