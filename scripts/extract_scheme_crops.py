"""
Extract table_scheme crops from images using the tab-seg YOLO model.
Usage:
    flowfigtabminer/bin/python scripts/extract_scheme_crops.py \
        --input_dir /path/to/images \
        --output_dir /path/to/output \
        [--conf 0.3]
"""
import os
import sys
import argparse
import glob
import cv2

sys.path.insert(0, os.getcwd())

CLASS_MAP = {
    0: "table_body",
    1: "table_caption",
    2: "table_note",
    3: "table_scheme",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="models/yolo11m-tab-seg-0209-white/runs/detect/train/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model_path)
    print(f"Loaded model: {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")) +
                       glob.glob(os.path.join(args.input_dir, "*.jpg")) +
                       glob.glob(os.path.join(args.input_dir, "*.jpeg")))

    print(f"Found {len(img_paths)} images in {args.input_dir}")

    saved = 0
    skipped = 0

    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [SKIP] Cannot read: {img_path}")
            skipped += 1
            continue

        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]

        results = model(img, imgsz=1024, conf=args.conf, verbose=False)[0]

        scheme_idx = 0
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if CLASS_MAP.get(cls_id) != "table_scheme":
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            out_name = f"{stem}_scheme_{scheme_idx}_conf{conf:.2f}.png"
            out_path = os.path.join(args.output_dir, out_name)
            cv2.imwrite(out_path, crop)
            print(f"  [SAVED] {out_name}  box=({x1},{y1},{x2},{y2})  conf={conf:.2f}")
            scheme_idx += 1
            saved += 1

        if scheme_idx == 0:
            print(f"  [NO SCHEME] {os.path.basename(img_path)}")
            skipped += 1

    print(f"\nDone. Saved {saved} scheme crops, {skipped} images had no scheme detected.")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
