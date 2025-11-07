# create_person_coco.py
import json
from pathlib import Path
from tqdm import tqdm

def filter_person(ann_path: Path, out_path: Path):
    data = json.loads(ann_path.read_text())
    anns = data['annotations']
    images = data['images']
    cats = data['categories']

    # person category id in COCO is 1 (verify if needed)
    person_id = 1

    # Keep only annotations where category_id == person_id
    person_anns = [a for a in anns if a['category_id'] == person_id]

    # find image ids that have person annotations
    img_ids = set(a['image_id'] for a in person_anns)
    person_images = [img for img in images if img['id'] in img_ids]

    # create new category list with single 'person' entry (optional: keep original id)
    person_cat = [c for c in cats if c['id'] == person_id]

    out = {
        'images': person_images,
        'annotations': person_anns,
        'categories': person_cat
    }
    out_path.write_text(json.dumps(out))
    print(f"Saved {len(person_images)} images and {len(person_anns)} annotations to {out_path}")

if __name__ == "__main__":
    base = Path("annotations")  # change if your folder differs
    filter_person(base / "instances_train2017.json", base / "instances_train2017_person.json")
    filter_person(base / "instances_val2017.json", base / "instances_val2017_person.json")
