"""
Tab-Scheme-Seg Parser
专门 YOLO 模型（yolo11nano，4类），检测 table scheme 图中的：
  - arrow           → 确定反应方向（反应物 / 产物分界线）
  - molecule        → MolNexTR 提取 SMILES
  - table-mark      → OCR 提取化合物代号（如 "4j", "2b"）
  - table-condition → OCR 提取反应条件文本

类别映射（模型训练顺序）：
  {0: 'arrow', 1: 'molecule', 2: 'table-condition', 3: 'table-mark'}

空间匹配：
  - mark → molecule：优先 x 轴重叠（mark 正下方），fallback 欧氏距离
  - arrow 分界线：确定每个 molecule 是 reactant / product / intermediate
"""
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from src.extraction.common.content_recognizer import ContentRecognizer

CLASS_MAP = {0: "arrow", 1: "molecule", 2: "table-condition", 3: "table-mark"}


class SchemeSegParser:
    def __init__(self, model_path=None, conf_threshold=0.3):
        # 各类别独立阈值：molecule/arrow 用高阈值，mark/condition 用低阈值
        self.conf_per_class = {
            "arrow":           conf_threshold,
            "molecule":        conf_threshold,
            "table-condition": 0.1,
            "table-mark":      0.1,
        }
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            _torch_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
            self.model.to(_torch_device)
            print(f"[SchemeSegParser] Loaded model: {model_path} (device: {_torch_device.upper()})")
        else:
            print(f"[SchemeSegParser] Model not found at {model_path}, step will be skipped.")
        self.content_recognizer = ContentRecognizer()  # MolNexTR
        self._ocr = None

    def _get_rec(self):
        if self._ocr is None:
            from src.extraction.common.ocr_backend import get_rec_instance
            self._ocr = get_rec_instance()
        return self._ocr

    def _ocr_crop(self, img, box):
        """对给定 bbox 区域做 OCR (rec-only)，返回合并文本"""
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return ""
        # 小图放大到至少 150px（short side），保证 OCR 准确
        h, w = crop.shape[:2]
        min_side = 150
        if min(h, w) < min_side:
            scale = max(min_side / h, min_side / w)
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        rec = self._get_rec()
        text, _conf = rec.recognize(crop)
        return text.strip()

    def _letterbox_pad(self, img, size=1024):
        """
        模型训练使用 1024 + 黑边 padding。
        左上角放置，右/下填黑，返回 (padded_img, scale)。
        不需要坐标反算（crop 直接从 padded 图取）。
        """
        h, w = img.shape[:2]
        scale = min(size / h, size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = cv2.resize(img, (new_w, new_h))
        return padded, scale

    def parse_scheme(self, scheme_image_path: str) -> dict:
        """
        解析单张 scheme 图。

        Returns:
            {
                "reactant_pool":  {"2a": "SMILES..."},
                "product_pool":   {"3b": "SMILES..."},
                "compound_pool":  {},   # 仅无箭头时有内容
                "conditions_text": "MeLi, -78°C, THF/Et2O, ..."
            }
        """
        empty = {"reactant_pool": {}, "product_pool": {}, "compound_pool": {}, "conditions_text": ""}
        if self.model is None:
            return empty

        img = cv2.imread(scheme_image_path)
        if img is None:
            return empty

        # ── Step 1: YOLO 推理（黑边 letterbox） ──────────────────────────────
        padded, scale = self._letterbox_pad(img, size=1024)
        # 用最低阈值推理，分拣时按类别各自过滤
        min_conf = min(self.conf_per_class.values())
        results = self.model(padded, conf=min_conf, imgsz=1024, verbose=False)[0]
        boxes = results.boxes

        print(f"   [SchemeSegParser] YOLO detected {len(boxes)} boxes (min_conf={min_conf})")
        molecules, arrows, marks, conditions = [], [], [], []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = CLASS_MAP.get(cls_id, "unknown")
            xyxy = box.xyxy[0].tolist()  # 坐标在 padded 1024×1024 空间
            conf = float(box.conf[0])
            # 按类别阈值过滤
            if conf < self.conf_per_class.get(cls_name, 0.3):
                continue
            print(f"      cls={cls_id}({cls_name}) conf={conf:.2f} box={[int(v) for v in xyxy]}")
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w_box = x2 - x1
            h_box = y2 - y1
            entry = {
                "box": xyxy, "cx": cx, "cy": cy,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": w_box, "h": h_box, "conf": conf
            }

            if cls_name == "molecule":
                molecules.append(entry)
            elif cls_name == "arrow":
                arrows.append(entry)
            elif cls_name == "table-mark":
                marks.append(entry)
            elif cls_name == "table-condition":
                conditions.append(entry)

        # ── Step 2: 为每个 molecule 提取 SMILES（MolNexTR，从 padded 图 crop） ──
        for mol in molecules:
            x1, y1, x2, y2 = [int(v) for v in mol["box"]]
            crop = padded[max(0, y1 - 5):y2 + 5, max(0, x1 - 5):x2 + 5]
            if crop.size == 0:
                mol["smiles"] = None
                continue
            try:
                smiles = self.content_recognizer.recognize_content(crop, "Structure")
                mol["smiles"] = smiles if smiles and smiles != "N/A" else None
            except Exception:
                mol["smiles"] = None

        # ── Step 3: OCR table-mark → 代号文本 ────────────────────────────────
        for mark in marks:
            mark["text"] = self._ocr_crop(padded, mark["box"]).strip()

        # ── Step 4: 箭头分析 → 确定 first_cx / last_cx ───────────────────────
        # 过滤竖向箭头（高 > 宽）
        h_arrows = [a for a in arrows if a["w"] >= a["h"]]

        if not h_arrows:
            role_cx_first = None
            role_cx_last = None
        elif len(h_arrows) == 1:
            role_cx_first = h_arrows[0]["cx"]
            role_cx_last = h_arrows[0]["cx"]
        else:
            role_cx_first = min(a["cx"] for a in h_arrows)
            role_cx_last = max(a["cx"] for a in h_arrows)

        # ── Step 5: 为每个 molecule 标记角色 ────────────────────────────────
        for mol in molecules:
            if role_cx_first is None:
                mol["role"] = "unknown"
            elif mol["cx"] < role_cx_first:
                mol["role"] = "reactant"
            elif mol["cx"] > role_cx_last:
                mol["role"] = "product"
            else:
                mol["role"] = "intermediate"  # 跳过，不加入任何 pool

        # ── Step 6: mark → molecule 匹配（x 轴对齐优先，fallback 欧氏距离） ──
        reactant_pool: dict = {}
        product_pool: dict = {}
        compound_pool: dict = {}

        for mark in marks:
            label = mark["text"]
            if not label:
                continue

            mark_w = mark["w"]
            mark_cx = mark["cx"]
            mark_cy = mark["cy"]
            mark_y1 = mark["y1"]

            # (a) 优先 x 轴对齐：x 重叠 > 30% mark 宽度，且 mark 在 molecule 下方
            x_aligned = []
            for mol in molecules:
                overlap_x = min(mark["x2"], mol["x2"]) - max(mark["x1"], mol["x1"])
                if overlap_x > 0.3 * mark_w and mark_y1 > mol["y1"]:
                    dist_y = abs(mark_cy - mol["cy"])
                    x_aligned.append((dist_y, mol))

            if x_aligned:
                x_aligned.sort(key=lambda t: t[0])
                best_mol = x_aligned[0][1]
            else:
                # (b) fallback：欧氏距离最小（< 300px）
                best_mol = None
                best_dist = float("inf")
                for mol in molecules:
                    dist = ((mark_cx - mol["cx"]) ** 2 + (mark_cy - mol["cy"]) ** 2) ** 0.5
                    if dist < best_dist and dist < 300:
                        best_dist = dist
                        best_mol = mol

            if best_mol is None or not best_mol.get("smiles"):
                continue

            role = best_mol["role"]
            smiles = best_mol["smiles"]
            print(f"   [SchemeSegParser] {label} ({role}) -> {smiles[:40]}...")

            if role == "reactant":
                reactant_pool[label] = smiles
            elif role == "product":
                product_pool[label] = smiles
            elif role == "unknown":
                compound_pool[label] = smiles
            # intermediate 跳过

        # ── Step 7: OCR table-condition → 合并条件文本 ───────────────────────
        condition_texts = []
        for cond in conditions:
            text = self._ocr_crop(padded, cond["box"])
            if text:
                condition_texts.append(text)
        conditions_text = "; ".join(condition_texts)

        return {
            "reactant_pool": reactant_pool,
            "product_pool": product_pool,
            "compound_pool": compound_pool,
            "conditions_text": conditions_text,
        }
