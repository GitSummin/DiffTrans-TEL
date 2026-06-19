"""
Stage 2b — Object Compositing into Aerial Backgrounds

Composites synthetic object cutouts (produced by extract_masks.py) onto real
aerial/satellite background images. For each composited scene the script saves:
  1. Composited RGB image        -- objects alpha-blended at annotated spots
  2. Inpainting boundary mask    -- seam region around each object for Stage 3
  3. YOLO annotation .txt file   -- normalised (class cx cy w h) per object

Configuration (defined in the BG_CONFIGS and CLASS_MAP dicts below):

  CLASS_MAP: maps a filename keyword to a YOLO class index.
    e.g. {"Buk-M2": 0, "S-400": 1}

  BG_CONFIGS: one entry per background image (keyed by filename prefix).
    Required key:
      "spots"        -- list of (x_pct, y_pct, angle_deg) tuples
                        x_pct, y_pct : normalised object centre position [0, 1]
                        angle_deg    : desired heading angle for PCA alignment
    Scale key (one of):
      "scale_range"  -- (min_pct, max_pct)  object width as fraction of image
                        width, sampled uniformly each placement
      "scale_pct"    -- float               fixed object width fraction

  Workflow to build BG_CONFIGS:
    Run annotate_spots.py on each background image to interactively pick
    placement spots, then paste the printed tuples here.

Usage:
  python insert_objects.py \
      --objects_dir  extracted_results/<class>/objects/ \
      --bg_dir       backgrounds/ \
      --output_dir   composites/ \
      --copies_per_bg 30
"""

import os
import sys
import random
import argparse

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# CLASS_MAP  -- filename keyword -> YOLO class ID
# Add or modify entries to match your object categories.
# ---------------------------------------------------------------------------
CLASS_MAP = {
    "Buk-M2":     0,
    "S-400":      1,
    "Pantsir":    2,
    "Tor":        3,
    "RS-24":      4,
    "Iskander":   5,
    "Bal":        6,
    "BM-30":      7,
    "ISDM":       8,
    "TOS":        9,
}

# ---------------------------------------------------------------------------
# BG_CONFIGS  -- one entry per background image
# Key   : filename prefix (without extension) of the background image
# Value : dict with "scale_range" (or "scale_pct") and "spots"
#
# spots format: (x_pct, y_pct, angle_deg)
#   x_pct, y_pct -- normalised centre position of the object [0.0, 1.0]
#   angle_deg    -- desired heading angle; use 0 for axis-aligned objects
#
# Run annotate_spots.py on each background to obtain these values.
# ---------------------------------------------------------------------------
BG_CONFIGS = {
    # --- Example: TEL vehicles on a large-vehicle background ---------------
    "P0085": {
        "scale_range": (0.02, 0.025),
        "spots": [
            (0.448, 0.341,  75), (0.608, 0.762, -104), (0.486, 0.497, 162),
            (0.354, 0.712,  56), (0.696, 0.149,   43), (0.575, 0.132, -123),
            (0.908, 0.485,  22), (0.742, 0.790,  -84),
        ],
    },
    "P1073": {
        "scale_range": (0.02, 0.025),
        "spots": [
            (0.374, 0.108, 111), (0.216, 0.090, 115), (0.291, 0.559, 127),
            (0.172, 0.821, -79), (0.199, 0.827, -77), (0.226, 0.834, 180),
            (0.231, 0.805,  52), (0.616, 0.600,  29), (0.515, 0.150, 113),
        ],
    },
    # --- Example: small objects on an aerial scene (angle = 0) -------------
    "P0042": {
        "scale_pct": 0.04,
        "spots": [
            (0.210, 0.620, 0), (0.682, 0.342, 0), (0.651, 0.307, 0),
            (0.489, 0.720, 0), (0.532, 0.693, 0), (0.518, 0.770, 0),
        ],
    },
    "P0160": {
        "scale_pct": 0.02,
        "spots": [
            (0.433, 0.373, 0), (0.439, 0.412, 0), (0.468, 0.233, 0),
            (0.519, 0.217, 0), (0.527, 0.431, 0), (0.731, 0.247, 0),
            (0.745, 0.339, 0), (0.759, 0.444, 0), (0.273, 0.562, 0),
        ],
    },
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def align_object_to_angle(pil_img, target_angle):
    """Rotate the object so its principal axis aligns with target_angle (degrees).

    Uses PCA on the alpha-channel pixel coordinates to estimate the object's
    current orientation, then rotates it to match the desired angle.
    A random 180-degree flip is applied with 50% probability because vehicles
    can face either direction along the same road.
    """
    arr   = np.array(pil_img)
    alpha = arr[:, :, 3]
    y_pts, x_pts = np.nonzero(alpha)
    if len(x_pts) == 0:
        return pil_img

    coords = np.vstack([x_pts, y_pts])
    cov    = np.cov(coords)
    _, evecs = np.linalg.eig(cov)
    principal = evecs[:, np.argmax(np.linalg.eig(cov)[0])]

    obj_angle    = np.degrees(np.arctan2(principal[1], principal[0]))
    rotation     = target_angle - obj_angle
    if random.choice([True, False]):
        rotation += 180

    return pil_img.rotate(-rotation, expand=True, resample=Image.BICUBIC)


def check_overlap(new_box, placed_boxes):
    """Return True if new_box intersects any box in placed_boxes."""
    nx1, ny1, nx2, ny2 = new_box
    for (px1, py1, px2, py2) in placed_boxes:
        if max(nx1, px1) < min(nx2, px2) and max(ny1, py1) < min(ny2, py2):
            return True
    return False


def composite_scene(bg_path, obj_paths, output_bg_path, output_mask_path,
                    config, class_map, max_objects=10):
    """Composite objects onto one background image.

    Produces: composited RGB image, inpainting boundary mask, and a list of
    YOLO annotation strings.

    The inpainting mask covers the bounding rectangle of each pasted object
    (with a 10% margin) but excludes the object pixels themselves, so Stage 3
    (MAT) fills only the boundary seam and not the object body.
    """
    bg_img = ImageOps.exif_transpose(Image.open(bg_path)).convert("RGBA")
    bg_w, bg_h = bg_img.size

    # Black canvas; the boundary region around each object will be drawn white
    bg_mask_img = Image.new("L", (bg_w, bg_h), 0)

    # Resolve scale: either fixed or sampled from a range
    if "scale_range" in config:
        scale_min, scale_max = config["scale_range"]
        use_fixed_scale = False
    else:
        fixed_scale = config["scale_pct"]
        use_fixed_scale = True

    available_spots = config["spots"].copy()
    random.shuffle(available_spots)

    num_to_place = min(max_objects, len(available_spots), len(obj_paths))
    chosen_objs  = random.sample(obj_paths, num_to_place)
    placed_boxes = []
    yolo_labels  = []
    success_count = 0

    for i in range(num_to_place):
        obj_path = chosen_objs[i]
        pct_x, pct_y, spot_angle = available_spots.pop()
        spot_cx = int(bg_w * pct_x)
        spot_cy = int(bg_h * pct_y)

        obj_img = Image.open(obj_path).convert("RGBA")

        # Determine target scale as fraction of image width, then add ±5% jitter
        scale_pct = (random.uniform(scale_min, scale_max)
                     if not use_fixed_scale else fixed_scale)
        scale_factor = (bg_w * scale_pct) / float(obj_img.width)
        jitter_scale = scale_factor * random.uniform(0.95, 1.05)
        jitter_angle = spot_angle + random.uniform(-3.0, 3.0)

        new_w = int(obj_img.width  * jitter_scale)
        new_h = int(obj_img.height * jitter_scale)
        if new_w <= 0 or new_h <= 0:
            continue

        resized_obj = obj_img.resize((new_w, new_h), Image.LANCZOS)
        aligned_obj = align_object_to_angle(resized_obj, jitter_angle)

        # Centre the object on the annotated spot with a small random offset
        paste_x = int(spot_cx - aligned_obj.width  / 2) + random.randint(-5, 5)
        paste_y = int(spot_cy - aligned_obj.height / 2) + random.randint(-5, 5)

        # Skip if the object would be placed partially outside the image
        if (paste_x < 0 or paste_y < 0
                or paste_x + aligned_obj.width  > bg_w
                or paste_y + aligned_obj.height > bg_h):
            continue

        new_box = (paste_x, paste_y,
                   paste_x + aligned_obj.width, paste_y + aligned_obj.height)
        if check_overlap(new_box, placed_boxes):
            continue

        # Alpha-blend the object onto the background
        bg_img.paste(aligned_obj, (paste_x, paste_y), mask=aligned_obj)

        # Build the inpainting mask and YOLO annotation from the alpha channel
        alpha = np.array(aligned_obj.split()[3])
        y_idx, x_idx = np.nonzero(alpha)
        if len(x_idx) > 0:
            x_min, x_max = int(np.min(x_idx)), int(np.max(x_idx))
            y_min, y_max = int(np.min(y_idx)), int(np.max(y_idx))

            # Assign YOLO class ID by matching filename keyword
            class_id = 0
            obj_name = Path(obj_path).name
            for keyword, mapped_id in class_map.items():
                if keyword in obj_name:
                    class_id = mapped_id
                    break

            # Tight bounding box in absolute pixel coordinates
            abs_x1 = paste_x + x_min
            abs_x2 = paste_x + x_max
            abs_y1 = paste_y + y_min
            abs_y2 = paste_y + y_max
            tight_w  = abs_x2 - abs_x1
            tight_h  = abs_y2 - abs_y1
            yolo_labels.append(
                f"{class_id} "
                f"{(abs_x1 + tight_w / 2.0) / bg_w:.6f} "
                f"{(abs_y1 + tight_h / 2.0) / bg_h:.6f} "
                f"{tight_w / bg_w:.6f} "
                f"{tight_h / bg_h:.6f}"
            )

            # Inpainting mask: bounding rectangle (with 10% margin) minus object pixels
            m  = max(5, int((x_max - x_min) * 0.1))
            lx1, ly1 = max(0, x_min - m), max(0, y_min - m)
            lx2, ly2 = min(aligned_obj.width, x_max + m), min(aligned_obj.height, y_max + m)
            local_mask      = np.zeros_like(alpha)
            local_mask[ly1:ly2, lx1:lx2] = 255
            local_mask[alpha > 127]       = 0   # exclude the object body
            bg_mask_img.paste(
                Image.fromarray(local_mask, 'L'),
                (paste_x, paste_y),
                mask=Image.fromarray(local_mask, 'L')
            )

        placed_boxes.append(new_box)
        success_count += 1

    if success_count > 0:
        bg_img.convert("RGB").save(output_bg_path)
        bg_mask_img.save(output_mask_path)
        return success_count, yolo_labels

    return 0, []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Composite object cutouts into aerial backgrounds with YOLO labels."
    )
    parser.add_argument("--objects_dir",    type=str, required=True,
                        help="Directory containing *_obj.png cutout files")
    parser.add_argument("--bg_dir",         type=str, required=True,
                        help="Directory containing background images")
    parser.add_argument("--output_dir",     type=str, default="composites",
                        help="Root output directory (images/, masks/, labels/ created inside)")
    parser.add_argument("--copies_per_bg",  type=int, default=30,
                        help="Number of augmented copies to generate per background image")
    args = parser.parse_args()

    obj_files = [str(p) for p in Path(args.objects_dir).glob("*_obj.png")]
    if not obj_files:
        print("[ERROR] No *_obj.png files found in:", args.objects_dir)
        sys.exit(1)

    for subdir in ("images", "masks", "labels"):
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    print(f"[INFO] {len(obj_files)} object cutout(s) found.")
    print(f"[INFO] Generating {args.copies_per_bg} composite(s) per background.")

    for bg_prefix, config in BG_CONFIGS.items():
        bg_path = None
        for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
            candidate = os.path.join(args.bg_dir, bg_prefix + ext)
            if os.path.exists(candidate):
                bg_path = candidate
                break
        if bg_path is None:
            continue

        print(f"[{Path(bg_path).name}] compositing {args.copies_per_bg} scene(s)...")
        num_spots = len(config["spots"])

        for i in range(args.copies_per_bg):
            out_img   = os.path.join(args.output_dir, "images", f"{bg_prefix}_aug_{i}.png")
            out_mask  = os.path.join(args.output_dir, "masks",  f"{bg_prefix}_aug_{i}_mask.png")
            out_label = os.path.join(args.output_dir, "labels", f"{bg_prefix}_aug_{i}.txt")

            # Randomise how many objects to place (3 to max, keep a few spots free)
            if num_spots >= 10:
                num_objs = random.randint(min(3, num_spots), num_spots - 3)
            else:
                num_objs = random.randint(min(3, num_spots), num_spots)

            count, yolo_data = composite_scene(
                bg_path, obj_files, out_img, out_mask,
                config, CLASS_MAP, max_objects=num_objs
            )
            if count > 0:
                with open(out_label, "w") as f:
                    f.write("\n".join(yolo_data))
                print(f"  [OK] aug_{i}: {count} object(s) placed")

    print(f"\n[DONE] Outputs saved to '{args.output_dir}' (images/, masks/, labels/)")
