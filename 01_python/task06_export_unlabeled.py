"""
task06_export_unlabeled.py — WM-811K 미분류(unlabeled) 웨이퍼맵 정리

LSWMD.pkl(WM-811K 원본, 811,457장) 안에는 우리가 쓰는 9개 클래스
(Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none)
어디에도 라벨링되지 못한 미분류 웨이퍼맵이 있다. 이 스크립트는:

  1. failureType 컬럼을 조사해 "미분류"를 정확히 식별하고,
  2. 미분류 웨이퍼맵 전체를 하나의 .npy(object 배열, 원본 크기 그대로)로 저장하고,
  3. 그중 일부를 무작위 샘플링해 PNG로 export한다(나중에 wafer_cnn_gui_batch.py로
     "모델이 미분류 데이터를 어떻게 예측하는지" 테스트하는 용도).

미분류 판정 기준:
  task02_data_preprocess.py 의 unpack(val) 방식(`val[0][0]`, 실패 시 '')과 동일한
  규칙을 적용한다. 확인 결과 미분류 행의 failureType 은 shape=(0,0) 인 빈 ndarray이며,
  라벨된 행은 shape=(1,1) 인 문자열 ndarray다. 즉 unpack() 결과가 '' 인 행 = 미분류.

PNG 변환 규칙(task02_data_preprocess.py의 preprocess()와 동일하게 맞춤):
  wmap(0/1/2 정수, 원본 해상도) → skimage.resize로 48×48 리사이즈(anti_aliasing=False,
  preserve_range=True) → /2.0 로 0~1 정규화 → *255 로 8bit 그레이스케일 PNG 저장.
  이렇게 하면 배치 GUI(wafer_cnn_gui_batch.py)의 image_to_bin()이 이 PNG를 다시
  48×48로 리사이즈(사실상 변화 없음, 이미 48×48)하고 /255*INPUT_SCALE(4)로 양자화
  했을 때, 학습에 쓰인 X.npy(0/0.5/1.0 값)와 동일한 스케일이 재현된다.

사용:
  cd 01_python && python task06_export_unlabeled.py [--n-samples 300] [--seed 0]
"""

import os
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize

_HERE = os.path.dirname(os.path.abspath(__file__))
PKL_PATH  = os.path.normpath(os.path.join(_HERE, "..", "00_data", "LSWMD.pkl"))
OUT_DIR   = os.path.normpath(os.path.join(_HERE, "..", "00_data", "unlabeled"))
NPY_PATH  = os.path.join(OUT_DIR, "unlabeled_all.npy")
PNG_DIR   = os.path.join(OUT_DIR, "png_samples")
IMG_SIZE  = (48, 48)

KNOWN_9 = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
           "Loc", "Near-full", "Random", "Scratch", "none"]


def unpack(val):
    """task02_data_preprocess.py 와 동일한 라벨 추출 규칙."""
    try:
        return val[0][0]
    except Exception:
        return ''


def render_png_array(wmap):
    """task02.preprocess()와 동일한 리사이즈+정규화 후, PNG 저장용 0~255 uint8로 변환."""
    arr = np.array(wmap, dtype=np.float32)
    resized = resize(arr, IMG_SIZE, anti_aliasing=False, preserve_range=True)
    normalized = resized / 2.0                      # 0/1/2 → 0~1.0
    img = np.clip(np.round(normalized * 255), 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser(description="WM-811K 미분류 웨이퍼맵 정리")
    ap.add_argument("--n-samples", type=int, default=500,
                     help="PNG로 export할 랜덤 샘플 수 (기본 300, 200~500 권장)")
    ap.add_argument("--seed", type=int, default=0, help="랜덤 시드")
    args = ap.parse_args()

    # ── 1단계: 데이터 구조 파악 ────────────────────────────────────
    print("=" * 60)
    print("[1단계] LSWMD.pkl 구조 파악")
    print("=" * 60)
    print(f"[로드] {PKL_PATH} 읽는 중... (약 2GB, 시간이 걸립니다)")
    df = pd.read_pickle(PKL_PATH)
    print(f"[로드] 전체 행 수: {len(df):,}")
    print(f"[컬럼] {list(df.columns)}")

    labels = df["failureType"].apply(unpack)
    vc = labels.value_counts(dropna=False)

    print("\n[라벨 고유값 및 개수]")
    for lab, cnt in vc.items():
        tag = "(9개 클래스)" if lab in KNOWN_9 else "(미분류 추정)"
        show = lab if lab != "" else "'' (빈 문자열)"
        print(f"  {show:20s}: {cnt:>8,}  {tag}")

    unlabeled_mask = (labels == "")
    n_unlabeled = int(unlabeled_mask.sum())
    print(f"\n[결과] 9개 클래스와 정확히 매칭: {sorted(set(labels.unique()) & set(KNOWN_9))}")
    print(f"[결과] 그 외(미분류) 값: {sorted(set(labels.unique()) - set(KNOWN_9))}")
    print(f"[결과] 미분류 데이터 개수: {n_unlabeled:,} / 전체 {len(df):,}")

    # ── 2단계: 폴더 정리 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[2단계] 미분류 데이터 저장")
    print("=" * 60)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PNG_DIR, exist_ok=True)

    unlabeled_df = df[unlabeled_mask].reset_index()  # 원본 인덱스를 'index' 컬럼으로 보존
    wafer_maps = unlabeled_df["waferMap"].to_numpy(dtype=object)

    print(f"[저장] {NPY_PATH} 에 원본 웨이퍼맵 {len(wafer_maps):,}개 저장 중 "
          f"(크기가 제각각이라 object 배열로 저장)...")
    np.save(NPY_PATH, wafer_maps, allow_pickle=True)
    print(f"[저장] 완료 - shape={wafer_maps.shape}, dtype={wafer_maps.dtype}")

    n_samples = min(args.n_samples, len(unlabeled_df))
    rng = np.random.default_rng(args.seed)
    sample_pos = rng.choice(len(unlabeled_df), size=n_samples, replace=False)
    sample_pos.sort()

    print(f"\n[export] PNG 샘플 {n_samples}장 생성 중 → {PNG_DIR}")
    saved = 0
    for pos in sample_pos:
        row = unlabeled_df.iloc[pos]
        orig_idx = int(row["index"])  # LSWMD.pkl 원본 행 인덱스
        try:
            png_arr = render_png_array(row["waferMap"])
            fname = f"idx{orig_idx:06d}.png"
            Image.fromarray(png_arr, mode="L").save(os.path.join(PNG_DIR, fname))
            saved += 1
        except Exception as e:
            print(f"  [스킵] 원본 idx {orig_idx}: {e}")

    print(f"[export] 완료 - {saved}장 저장")

    # ── 3단계: 요약 ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[3단계] 요약")
    print("=" * 60)
    print(f"전체 라벨 종류 및 개수:")
    for lab, cnt in vc.items():
        show = lab if lab != "" else "(미분류)"
        print(f"  {show:12s}: {cnt:,}")
    print(f"\n미분류 판단 기준: failureType 이 shape=(0,0)인 빈 ndarray"
          f" (unpack() 결과가 빈 문자열 '')")
    print(f".npy 저장: {NPY_PATH}")
    print(f"  shape={wafer_maps.shape}, dtype={wafer_maps.dtype}, 개수={len(wafer_maps):,}")
    print(f"PNG 샘플: {saved}장 → {PNG_DIR}")


if __name__ == "__main__":
    main()
