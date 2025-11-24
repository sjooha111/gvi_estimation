import os, gzip, torch, numpy as np, pandas as pd
from model.Revit import ReViT

# ====== 설정 ======
USE_NORMALIZATION = True  # <- 정규화 끄고 싶으면 False로 바꾸거나 관련 부분 삭제

# --- Config (간소화 버전) ---
class Config:
    class DATA: crop_size = 224
    class MODEL:
        num_classes = 1
        dropout_rate = 0.1
        head_act = None
    class ReViT:
        mode = "conv"
        pool_first = False
        patch_kernel = [16, 16]
        patch_stride = [16, 16]
        patch_padding = [0, 0]
        embed_dim = 768
        num_heads = 12
        mlp_ratio = 4
        qkv_bias = True
        drop_path = 0.2
        depth = 12
        dim_mul = []
        head_mul = []
        pool_qkv_kernel = []
        pool_kv_stride_adaptive = []
        pool_q_stride = []
        zero_decay_pos = False
        use_abs_pos = True
        use_rel_pos = False
        rel_pos_zero_init = False
        residual_pooling = False
        dim_mul_in_att = False
        alpha = True
        visualize = True
        cls_embed_on = False


def load_npy_gz(path):
    """압축된 npy.gz 파일을 읽어서 torch tensor로 변환"""
    with gzip.open(path, "rb") as f:
        arr = np.load(f)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)  # [1,C,H,W]


def load_minmax_stats(stats_path, device):
    """
    normalization_stats.txt에서 Overall Max/Min 값을 읽어서
    [1, C, 1, 1] shape의 tensor로 반환
    기대 형식:
        Overall Max:
        1.22,2.15, ...
        Overall Min:
        0.002,0.002, ...
    혹은 한 줄에 콜론 뒤에 바로 값들이 있어도 동작하도록 처리
    """
    if not os.path.exists(stats_path):
        print(f"[WARN] normalization stats file not found: {stats_path}")
        return None, None

    with open(stats_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    max_vals = None
    min_vals = None

    for line in lines:
        lower = line.lower()
        if "overall max" in lower:
            # "Overall Max:" 라인일 수도 있고, "Overall Max: 1.22,..." 한 줄일 수도 있음
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                if after:
                    max_vals = [float(x) for x in after.split(",")]
            # 값이 다음 줄에 있는 경우 처리 (예: 다음 줄이 숫자 리스트)
        elif (max_vals is None and
              all(c.isdigit() or c in ".,- " for c in line) and
              "overall" not in lower):
            # 숫자 리스트처럼 보이면 max 또는 min일 수 있음 (앞에서 max 라인을 본 후일 것)
            # 하지만 이건 아래에서 다시 처리하므로 pass
            pass

    # 위에서 max를 못 읽은 경우, "Overall Max:" 다음 줄 방식 처리
    if max_vals is None:
        for i, line in enumerate(lines):
            if "overall max" in line.lower() and i + 1 < len(lines):
                max_vals = [float(x) for x in lines[i + 1].split(",")]
                break

    # Min도 동일하게 처리
    for line in lines:
        lower = line.lower()
        if "overall min" in lower:
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                if after:
                    min_vals = [float(x) for x in after.split(",")]

    if min_vals is None:
        for i, line in enumerate(lines):
            if "overall min" in line.lower() and i + 1 < len(lines):
                min_vals = [float(x) for x in lines[i + 1].split(",")]
                break

    if max_vals is None or min_vals is None:
        print("[WARN] Failed to parse Overall Max/Min from stats file.")
        return None, None

    if len(max_vals) != len(min_vals):
        print("[WARN] len(max_vals) != len(min_vals). Check stats file.")
        return None, None

    max_t = torch.tensor(max_vals, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    min_t = torch.tensor(min_vals, dtype=torch.float32, device=device).view(1, -1, 1, 1)

    print(f"[INFO] Loaded normalization stats for {len(max_vals)} channels from {stats_path}")
    print(f"       Max (first 5): {max_vals[:5]}")
    print(f"       Min (first 5): {min_vals[:5]}")
    return min_t, max_t

def normalize_input(x, ch_min, ch_max):
    """
    x: [B, C, H, W]
    ch_min, ch_max: [1, K, 1, 1]  (K <= C 라고 가정)
    앞의 K개 채널만 (x - min) / (max - min) 으로 정규화하고,
    나머지 C-K 채널은 원본 그대로 둔다.
    """
    if ch_min is None or ch_max is None:
        return x

    B, C, H, W = x.shape
    K = ch_min.shape[1]

    if K > C:
        print(
            f"[WARN] Stats have {K} channels but input has only {C} channels. "
            f"Skipping normalization."
        )
        return x

    # 복사본 만들어서 앞 K채널만 수정
    x_norm = x.clone()

    denom = ch_max[:, :K] - ch_min[:, :K]
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)

    x_k = x[:, :K]  # 앞 K개 채널
    x_k = (x_k - ch_min[:, :K]) / denom
    x_k = torch.clamp(x_k, 0.0, 1.0)

    x_norm[:, :K] = x_k

    if C > K:
        print(f"[INFO] Normalized first {K} channels out of {C}. "
              f"Remaining {C-K} channels are left unchanged.")

    return x_norm

def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(repo_root, "data", "RS_sample")
    ckpt_path = os.path.join(repo_root, "checkpoint", "Revit_checkpoint.bin")
    # truth_csv = os.path.join(repo_root, "data", "dataset.csv")
    truth_csv = os.path.join(repo_root, "data", "dataset_mapo.csv")
    stats_path = os.path.join(repo_root, "normalization_stats.txt")  # <- 여기서 파일 사용
    # output_dir = os.path.join(repo_root, "output")
    output_dir = os.path.join(repo_root, "output_mapo")
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "prediction_with_truth.csv")
    metrics_path = os.path.join(output_dir, "metrics_nm.txt")


    # --- Load dataset.csv (모든 컬럼 유지) ---
    df_truth = pd.read_csv(truth_csv, encoding="cp949")

    # --- Load model ---
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ReViT(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.eval()

    # --- Load normalization stats (optional) ---
    ch_min, ch_max = None, None
    if USE_NORMALIZATION:
        ch_min, ch_max = load_minmax_stats(stats_path, device)
        if ch_min is None or ch_max is None:
            print("[WARN] Normalization disabled because stats could not be loaded.")
            # 실패했으면 그냥 정규화 없이 진행
            # (원하면 여기서 프로그램 종료해도 됨)
            # exit(1)

    gx = torch.tensor([[0.5, 0.5]], dtype=torch.float32).to(device)
    preds = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npy.gz"):
            continue

        # 파일명에서 sid 추출
        sid = int(os.path.splitext(fname.split(".")[0])[0])
        x = load_npy_gz(os.path.join(data_dir, fname)).to(device)  # [1, C, H, W]

        # --- 여기서 normalization 적용 ---
        if USE_NORMALIZATION and ch_min is not None and ch_max is not None:
            x = normalize_input(x, ch_min, ch_max)

        with torch.no_grad():
            pred = model(x, gx).squeeze().item()

        preds.append((sid, pred))

        # 정답값이 있으면 출력
        row = df_truth[df_truth["sid"] == sid]
        if not row.empty:
            gvi_true = row.iloc[0]["gvi"]
            print(f"{fname} (sid={sid}) → Pred: {pred:.4f}, Truth: {gvi_true:.4f}")
        else:
            print(f"{fname} (sid={sid}) → Pred: {pred:.4f}, Truth: 없음")

    # --- DataFrame 병합 ---
    df_pred = pd.DataFrame(preds, columns=["sid", "predicted_GVI"])
    df_merge = pd.merge(df_truth, df_pred, on="sid", how="left")

    # === 성능 평가 (정답 gvi와 예측값이 모두 있는 샘플만 사용) ===
    valid = df_merge.dropna(subset=["gvi", "predicted_GVI"])
    if len(valid) > 0:
        y_true = valid["gvi"].to_numpy()
        y_pred = valid["predicted_GVI"].to_numpy()

        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        # 분산이 0인 경우(모든 정답이 같은 값) R^2 정의 불가 → np.nan 처리
        denom = np.sum((y_true - y_true.mean()) ** 2)
        if denom == 0:
            r2 = np.nan
        else:
            r2 = 1 - np.sum((y_true - y_pred) ** 2) / denom

        print("\n=== Evaluation on test data (samples with ground-truth GVI) ===")
        print(f"#Samples: {len(valid)}")
        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MSE : {mse:.4f}")
        print(f"R^2 : {r2:.4f}" if not np.isnan(r2) else "R^2 : NaN (constant ground truth)")

        # 결과를 파일로도 저장
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("Evaluation on test data (samples with ground-truth GVI)\n")
            f.write(f"#Samples: {len(valid)}\n")
            f.write(f"MAE : {mae:.6f}\n")
            f.write(f"RMSE: {rmse:.6f}\n")
            f.write(f"MSE : {mse:.6f}\n")
            f.write(f"R^2 : {r2:.6f}\n" if not np.isnan(r2) else "R^2 : NaN (constant ground truth)\n")
        print(f"\n📄 Saved metrics to: {metrics_path}")
    else:
        print("\n⚠ 평가할 수 있는 (gvi+예측) 샘플이 없습니다.")

    # --- Save (예측 포함 전체 테이블) ---
    df_merge.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved results (all dataset columns + predicted_GVI) to:\n{out_csv}")


if __name__ == "__main__":
    main()
