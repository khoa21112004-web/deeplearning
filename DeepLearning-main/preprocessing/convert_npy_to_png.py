import argparse
from pathlib import Path

import numpy as np
from PIL import Image


SPLITS = ("train", "valid", "test")
PLANES = ("axial", "coronal", "sagittal")


def minmax_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - vmin) / (vmax - vmin)
    arr = np.clip(arr * 255.0, 0.0, 255.0)
    return arr.astype(np.uint8)


def save_2d_png(src_npy: Path, dst_png: Path) -> None:
    arr = np.load(src_npy)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {src_npy}, got shape={arr.shape}")
    img = Image.fromarray(minmax_to_uint8(arr), mode="L")
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_png)


def save_3d_png_slices(src_npy: Path, dst_dir: Path) -> int:
    arr = np.load(src_npy)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array in {src_npy}, got shape={arr.shape}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i in range(arr.shape[0]):
        slice_2d = minmax_to_uint8(arr[i])
        out_path = dst_dir / f"{src_npy.stem}_{i:03d}.png"
        Image.fromarray(slice_2d, mode="L").save(out_path)
        count += 1
    return count


def convert_dataset(input_root: Path, output_root: Path, keep_2d_name: bool) -> None:
    npy_files = 0
    png_files = 0

    for split in SPLITS:
        for plane in PLANES:
            src_dir = input_root / split / plane
            if not src_dir.exists():
                print(f"[WARN] Missing folder, skipped: {src_dir}")
                continue

            for src_npy in sorted(src_dir.glob("*.npy")):
                npy_files += 1
                rel_dir = Path(split) / plane

                arr = np.load(src_npy, mmap_mode="r")
                if arr.ndim == 2:
                    if keep_2d_name:
                        dst_png = output_root / rel_dir / f"{src_npy.stem}.png"
                    else:
                        dst_png = output_root / rel_dir / f"{src_npy.stem}_000.png"
                    save_2d_png(src_npy, dst_png)
                    png_files += 1
                elif arr.ndim == 3:
                    dst_dir = output_root / rel_dir / src_npy.stem
                    png_files += save_3d_png_slices(src_npy, dst_dir)
                else:
                    raise ValueError(f"Unsupported ndim={arr.ndim} in file: {src_npy}")

    print(f"[DONE] Converted {npy_files} npy files into {png_files} png files.")
    print(f"[OUT] {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MRNet-style .npy files to .png")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(r"F:\NgDuyLinh\datasetDeep\data"),
        help="Root folder containing train/valid/test",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"F:\NgDuyLinh\datasetDeep\data_png"),
        help="Output root folder for PNG files",
    )
    parser.add_argument(
        "--keep-2d-name",
        action="store_true",
        help="For 2D npy, keep filename as <name>.png instead of <name>_000.png",
    )
    args = parser.parse_args()

    convert_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        keep_2d_name=args.keep_2d_name,
    )


if __name__ == "__main__":
    main()
