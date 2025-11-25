import os
import gzip
import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

logger = logging.getLogger(__name__)

class MyDatasetPro(Dataset):
    def __init__(self, img_dir, sids, positions, targets=None, events=None):
        """
        sids      : (N,) 문자열 sid
        positions : (N,2) float32 (정규화된 lon, lat)
        targets   : (N,) float32 또는 None (gvi / extended_gvi)
        events    : (N,) int64 또는 None (final_class_index)
        """
        self.img_dir = img_dir
        self.sids = np.asarray(sids)
        self.positions = np.asarray(positions, dtype=np.float32)

        self.targets = None if targets is None else np.asarray(targets, dtype=np.float32)
        self.events = None if events is None else np.asarray(events, dtype=np.int64)

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, index):
        sid = str(self.sids[index])
        sid_prefix = sid.split("_19ch")[0]
        file_name = f"{sid_prefix}_19ch.npz"
        img_path = os.path.join(self.img_dir, file_name)

        npz = np.load(img_path)
        img_np = npz["combined_data"]   # ← 네가 확인해준 키 이름
        img = torch.tensor(img_np, dtype=torch.float32)

        pos = torch.tensor(self.positions[index], dtype=torch.float32)

        # 반환 형태는 가지고 있는 타깃에 따라 자동 결정
        if self.targets is not None and self.events is not None:
            veg = torch.tensor([self.targets[index]], dtype=torch.float32)
            event = torch.tensor(self.events[index], dtype=torch.long)
            return img, pos, veg, event

        elif self.targets is not None:
            veg = torch.tensor([self.targets[index]], dtype=torch.float32)
            return img, pos, veg

        elif self.events is not None:
            event = torch.tensor(self.events[index], dtype=torch.long)
            return img, pos, event

        else:
            # 타깃이 전혀 없는 경우는 거의 없겠지만, 방어용
            return img, pos



# class MyDatasetPro(Dataset):
#     def __init__(self, img_dir, sids, positions, targets):
#         """
#         img_dir   : 이미지 npz 파일이 있는 폴더
#         sids      : (N,) 문자열 배열
#         positions : (N,2) float32 numpy 배열
#         targets   : (N,) float32 numpy 배열
#         """
#         self.img_dir = img_dir
#         self.sids = np.asarray(sids)
#         self.positions = np.asarray(positions, dtype=np.float32)
#         self.targets = np.asarray(targets, dtype=np.float32)

#     def __len__(self):
#         return len(self.sids)

#     def __getitem__(self, index):
#         # -----------------------------
#         # 1) sid → 파일명 변환
#         # -----------------------------
#         sid = str(self.sids[index])
#         sid_prefix = sid.split("_19ch")[0]     # <-- 사용자가 알려준 규칙
#         file_name = f"{sid_prefix}_19ch.npz"
#         img_path = os.path.join(self.img_dir, file_name)

#         # -----------------------------
#         # 2) npz 파일 로드
#         # -----------------------------

#         npz = np.load(img_path)
#         img_np = npz["combined_data"]
#         img = torch.tensor(img_np, dtype=torch.float32)

#         # -----------------------------
#         # 3) 위치 / GVI label
#         # -----------------------------
#         pos = torch.tensor(self.positions[index], dtype=torch.float32)
#         veg = torch.tensor([self.targets[index]], dtype=torch.float32)

#         return img, pos, veg

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
    """
    train/val DataLoader 생성.
    task_type 에 따라 reg / cls / mtl 모두 지원.
      - reg : (img, pos, veg)
      - cls : (img, pos, event)
      - mtl : (img, pos, veg, event)
    """
    img_dir = args.img_dir
    data = pd.read_csv(args.dataset_csv)

    task = getattr(args, "task_type", "reg")          # 'reg', 'cls', 'mtl'
    target_col = getattr(args, "target_col", "gvi")   # 회귀 타깃 컬럼 이름

    # ---------- 1) train / val split ----------
    train_data = data[data["split"] == 0].copy()
    val_data   = data[data["split"] == 1].copy()

    # ---------- 2) 공통: sid / 좌표 준비 ----------
    lon_min, lon_max = 102.0, 130.5
    lat_min, lat_max = 22.0, 46.5

    def prepare_split(df):
        sids = df["sid"].astype(str).values

        lon = df["longitude"].astype("float32").values
        lat = df["latitude"].astype("float32").values

        lon_norm = (lon - lon_min) / (lon_max - lon_min)
        lat_norm = (lat - lat_min) / (lat_max - lat_min)
        positions = np.stack([lon_norm, lat_norm], axis=1).astype("float32")

        targets = None
        events  = None

        # 회귀 타깃
        if task in ["reg", "mtl"]:
            if target_col not in df.columns:
                raise KeyError(
                    f"target_col '{target_col}' not found in CSV. "
                    f"Available columns: {list(df.columns)}"
                )
            targets = df[target_col].astype("float32").values

        # 분류 라벨 (이벤트 클래스)
        if task in ["cls", "mtl"]:
            if "final_class_index" not in df.columns:
                raise KeyError("'final_class_index' not in CSV columns")
            events = df["final_class_index"].astype("int64").values

        return sids, positions, targets, events

    train_sids, train_pos, train_targets, train_events = prepare_split(train_data)
    val_sids,   val_pos,   val_targets,   val_events   = prepare_split(val_data)

    # ---------- 3) Dataset ----------
    train_dataset = MyDatasetPro(
        img_dir=img_dir,
        sids=train_sids,
        positions=train_pos,
        targets=train_targets,
        events=train_events,
    )

    val_dataset = MyDatasetPro(
        img_dir=img_dir,
        sids=val_sids,
        positions=val_pos,
        targets=val_targets,
        events=val_events,
    )

    # ---------- 4) DataLoader ----------
    num_workers = getattr(args, "num_workers", 2)

    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, eval_loader

    
def get_test_loader(args):
    """
    split == 2 인 test 데이터로부터 DataLoader 생성.
    task_type 에 따라 reg/cls/mtl 모두 지원.
    """
    img_dir = args.img_dir
    data = pd.read_csv(args.dataset_csv)

    task = getattr(args, "task_type", "reg")       # 'reg', 'cls', 'mtl'
    target_col = getattr(args, "target_col", "gvi")  # 회귀 타깃 컬럼: gvi 또는 extended_gvi

    # ------------- 1) test split 선택 -------------
    test_data = data[data["split"] == 2].copy()

    # ------------- 2) sid / 좌표 -------------
    test_sids = test_data["sid"].astype(str).values

    # lon/lat 정규화 (train/val 에서 쓰던 값과 꼭 동일해야 함)
    lon = test_data["longitude"].astype("float32").values
    lat = test_data["latitude"].astype("float32").values

    # 예전에 썼던 범위 재사용 (필요시 수정)
    lon_min, lon_max = 102.0, 130.5
    lat_min, lat_max = 22.0, 46.5

    lon_norm = (lon - lon_min) / (lon_max - lon_min)
    lat_norm = (lat - lat_min) / (lat_max - lat_min)
    test_pos = np.stack([lon_norm, lat_norm], axis=1).astype("float32")

    # ------------- 3) 타깃 생성 (reg / cls / mtl) -------------
    test_targets = None
    test_events = None

    if task in ["reg", "mtl"]:
        if target_col not in test_data.columns:
            raise KeyError(
                f"target_col '{target_col}' not found in CSV. "
                f"Available columns: {list(test_data.columns)}"
            )
        test_targets = test_data[target_col].astype("float32").values

    if task in ["cls", "mtl"]:
        if "final_class_index" not in test_data.columns:
            raise KeyError("'final_class_index' not in CSV columns")
        test_events = test_data["final_class_index"].astype("int64").values

    # ------------- 4) Dataset & DataLoader -------------
    test_dataset = MyDatasetPro(
        img_dir=img_dir,
        sids=test_sids,
        positions=test_pos,
        targets=test_targets,
        events=test_events,
    )

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.eval_batch_size,
        num_workers=getattr(args, "num_workers", 2),
        pin_memory=True,
    )

    return test_loader


