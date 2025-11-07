import json
from pathlib import Path
from tqdm import tqdm

# CONFIG
COCO_JSON = Path("datasets/coco_person_mini/instances_train2017_person_mini.json")
IMG_DIR   = Path("datasets/coco_person_mini/images")
LABEL_DIR = Path("datasets/coco_person_mini/labels")

def convert():
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    data = json.load(open(COCO_JSON))
    anns_by_img = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    
    img_map = {img["id"]: img for img in data["images"]}
    print(f"Converting {len(img_map)} images...")

    for img_id, img in tqdm(img_map.items()):
        w, h = img["width"], img["height"]
        file_name = Path(img["file_name"]).stem
        label_path = LABEL_DIR / f"{file_name}.txt"
        lines = []

        for ann in anns_by_img.get(img_id, []):
            # COCO bbox = [x_min, y_min, width, height]
            x, y, bw, bh = ann["bbox"]
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            w_rel = bw / w
            h_rel = bh / h
            # class id = 0 because we have only 'person'
            lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_rel:.6f} {h_rel:.6f}")

        if lines:
            label_path.write_text("\n".join(lines))

    print(f"✅ Conversion done — YOLO labels saved in: {LABEL_DIR}")

if __name__ == "__main__":
    convert()
