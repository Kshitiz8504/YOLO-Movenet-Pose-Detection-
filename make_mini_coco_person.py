import json
import random
import time
import requests
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIGURATION ----------------
COCO_ANN = Path("datasets/coco/annotations/instances_train2017.json")  # Path to COCO annotation file
OUT_DIR = Path("datasets/coco_person_mini")                            # Output folder for mini dataset
NUM_IMAGES = 5000                                                      # Number of person images to download (adjust if needed)
MAX_RETRIES = 3                                                        # Retry count for failed downloads
TIMEOUT = 15                                                           # Seconds before a single request times out
# ------------------------------------------------


def download_image(url: str, dest_path: Path) -> bool:
    """Download an image with retry logic and timeout handling."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, stream=True, timeout=TIMEOUT)
            if r.status_code == 200:
                dest_path.write_bytes(r.content)
                return True
            else:
                print(f"⚠️ HTTP {r.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Error downloading {url} (attempt {attempt + 1}): {e}")
            time.sleep(2)  # brief wait before retry
    print(f"❌ Skipping {url} after {MAX_RETRIES} failed attempts.")
    return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = OUT_DIR / "images"
    images_dir.mkdir(exist_ok=True)
    ann_out = OUT_DIR / "instances_train2017_person_mini.json"

    # --- Load COCO annotations ---
    print("📂 Loading COCO annotations...")
    with open(COCO_ANN, "r") as f:
        data = json.load(f)

    # --- Filter to only 'person' category ---
    print("🔍 Filtering annotations for 'person'...")
    anns = [a for a in data["annotations"] if a["category_id"] == 1]
    img_ids = list({a["image_id"] for a in anns})
    random.shuffle(img_ids)
    img_ids = img_ids[:NUM_IMAGES]

    imgs = [i for i in data["images"] if i["id"] in img_ids]
    anns = [a for a in anns if a["image_id"] in img_ids]
    cats = [c for c in data["categories"] if c["id"] == 1]

    mini_data = {"images": imgs, "annotations": anns, "categories": cats}
    with open(ann_out, "w") as f:
        json.dump(mini_data, f)
    print(f"✅ Saved mini annotation file: {ann_out}")

    # --- Download each image safely ---
    base_url = "http://images.cocodataset.org/train2017/"
    print(f"🌐 Downloading {len(imgs)} images from COCO servers...")

    success_count = 0
    for im in tqdm(imgs, desc="Downloading images"):
        url = base_url + im["file_name"]
        dest = images_dir / im["file_name"]
        if dest.exists():
            continue  # skip already downloaded images
        if download_image(url, dest):
            success_count += 1

    print(f"\n✅ Download complete: {success_count}/{len(imgs)} images saved to {images_dir}")


if __name__ == "__main__":
    main()
