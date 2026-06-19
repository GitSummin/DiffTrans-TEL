"""
Stage 2a — Object Mask Extraction

Extracts clean RGBA cutout images and binary alpha masks from synthetic object
images generated on a pure-white background (Stage 1 output).

Algorithm (per image):
  1. Threshold at near-white (>= 245) to isolate the background.
  2. Apply morphological closing to fill internal white regions
     (e.g. white-painted rooftops, windows).
  3. Fit a polygon to the largest connected component using approxPolyDP
     (epsilon = 0.8% of perimeter) to turn jagged edges into clean lines.
  4. Crop tightly to the bounding box with a configurable margin.
  5. Save: (a) an RGBA PNG with a transparent background, and
           (b) a grayscale binary mask PNG for compositing in Stage 2b.

Usage:
  python extract_masks.py \
      --input_dir  <path-to-generated-images> \
      --output_dir extracted_results/          \
      --scale      1.0
"""

import cv2
import numpy as np
import os
import argparse
import sys
from pathlib import Path
from PIL import Image


def get_solid_object_mask_poly(image_bgr, thresh=245, pad=30, close_ksize=15):
    """Return a clean filled polygon mask for the foreground object.

    Handles hollow interiors and jagged edges by combining morphological
    closing with polygon approximation.
    """
    # Step 1: Add border padding so objects touching the image edge are not clipped
    padded = cv2.copyMakeBorder(
        image_bgr, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

    # Step 2: Strict white-background threshold (all channels >= thresh)
    lower_white = np.array([thresh, thresh, thresh], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    bg_mask = cv2.inRange(padded, lower_white, upper_white)
    fg_mask = cv2.bitwise_not(bg_mask)

    # Step 3: Morphological closing bridges internal white gaps inside the object
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    closed_fg = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Step 4: Keep only the largest connected component
    contours, _ = cv2.findContours(closed_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(image_bgr[:, :, 0]), None
    biggest_contour = max(contours, key=cv2.contourArea)

    # Step 5: Polygon approximation removes noise smaller than 0.8% of the perimeter,
    #         converting jagged edges into clean straight-line segments
    peri = cv2.arcLength(biggest_contour, True)
    epsilon = 0.008 * peri
    approx_contour = cv2.approxPolyDP(biggest_contour, epsilon, True)

    # Step 6: Rasterise the smoothed polygon into a filled binary mask
    solid_mask = np.zeros_like(closed_fg)
    cv2.drawContours(solid_mask, [approx_contour], -1, 255, cv2.FILLED)

    # Step 7: Remove the padding added in Step 1
    h, w = closed_fg.shape
    solid_mask = solid_mask[pad:h - pad, pad:w - pad]

    # Step 8: Smooth the mask boundary with a 3x3 Gaussian blur then re-threshold
    solid_mask = cv2.GaussianBlur(solid_mask, (3, 3), 0)
    _, solid_mask = cv2.threshold(solid_mask, 127, 255, cv2.THRESH_BINARY)

    final_contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_contour = max(final_contours, key=cv2.contourArea) if final_contours else approx_contour

    return solid_mask, final_contour


def process_and_extract(input_path, output_obj_path, output_mask_path,
                        scale_factor=1.0, margin_factor=0.1):
    """Extract one object cutout and its mask from a white-background image."""
    image_bgr = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"[ERROR] Cannot read image: {input_path}")
        return False

    solid_mask, biggest_contour = get_solid_object_mask_poly(image_bgr)
    if biggest_contour is None:
        print(f"[WARN] No foreground object found: {input_path}")
        return False

    # Compute bounding box and apply margin
    x, y, w, h = cv2.boundingRect(biggest_contour)
    margin = int(min(w, h) * margin_factor)
    h_img, w_img = image_bgr.shape[:2]
    x1 = max(x - margin, 0)
    y1 = max(y - margin, 0)
    x2 = min(x + w + margin, w_img)
    y2 = min(y + h + margin, h_img)

    # Crop to bounding box
    cropped_bgr  = image_bgr[y1:y2, x1:x2]
    cropped_mask = solid_mask[y1:y2, x1:x2]

    # Optional rescaling
    if scale_factor != 1.0:
        new_w = int(cropped_bgr.shape[1] * scale_factor)
        new_h = int(cropped_bgr.shape[0] * scale_factor)
        if new_w > 0 and new_h > 0:
            cropped_bgr  = cv2.resize(cropped_bgr,  (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            cropped_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Build RGBA image: background alpha = 0, foreground alpha = mask value
    cropped_bgra = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
    cropped_bgra[..., 3] = cropped_mask

    obj_pil  = Image.fromarray(cv2.cvtColor(cropped_bgra, cv2.COLOR_BGRA2RGBA), 'RGBA')
    mask_pil = Image.fromarray(cropped_mask, 'L')

    # Save outputs
    os.makedirs(os.path.dirname(output_obj_path),  exist_ok=True)
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    output_obj_path  = str(Path(output_obj_path).with_suffix('.png'))
    output_mask_path = str(Path(output_mask_path).with_suffix('.png'))
    obj_pil.save(output_obj_path,  format="PNG")
    mask_pil.save(output_mask_path, format="PNG")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract RGBA object cutouts and binary masks from white-background images."
    )
    parser.add_argument("--input_dir",  type=str, required=True,
                        help="Directory containing white-background PNG/JPG images")
    parser.add_argument("--output_dir", type=str, default="extracted_results",
                        help="Root directory for extracted objects and masks")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Resize scale applied to each crop (1.0 = original size)")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    target_output_dir = os.path.join(args.output_dir, input_path.name)

    image_files  = [str(p) for p in input_path.rglob("*.png")]
    image_files += [str(p) for p in input_path.rglob("*.jpg")]

    if not image_files:
        print(f"[WARN] No images found in '{args.input_dir}'.")
        sys.exit()

    print(f"[INFO] Processing {len(image_files)} images (scale={args.scale})")
    print(f"[INFO] Output directory: {target_output_dir}")

    success_count = 0
    for file_path in image_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        out_obj  = os.path.join(target_output_dir, "objects", f"{base_name}_obj.png")
        out_mask = os.path.join(target_output_dir, "masks",   f"{base_name}_mask.png")
        if process_and_extract(file_path, out_obj, out_mask, scale_factor=args.scale):
            success_count += 1
            print(f"  [OK] {base_name}")

    print(f"\n[DONE] {success_count}/{len(image_files)} images extracted -> {target_output_dir}")
