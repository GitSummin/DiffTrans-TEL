"""
Stage 2b — Interactive Spot Annotator

An interactive OpenCV tool for annotating object placement spots on a
background image. The printed output can be pasted directly into the
BG_CONFIGS dict in insert_objects.py.

Two modes (controlled by --with-angle):

  Position-only mode (default, --with-angle not set):
    Single left-click to place a spot.
    Output: (x_pct, y_pct, 0),
    Use this for objects whose orientation does not matter (angle = 0).

  Position + angle mode (--with-angle):
    Click 1: centre of the object placement spot  (red dot drawn)
    Click 2: direction the object's front should face (green line drawn)
    Output: (x_pct, y_pct, angle_deg),
    Use this for vehicles whose heading must match the road/parking geometry.

Coordinates are expressed as fractions of the image width/height [0.0, 1.0]
so they remain valid regardless of how the image is displayed or resized.

Usage:
  # Position-only (e.g. aircraft on an apron)
  python annotate_spots.py --img backgrounds/P0042.jpg

  # Position + angle (e.g. TEL vehicles on a road)
  python annotate_spots.py --img backgrounds/P0085.jpg --with-angle

Press any key to quit.
"""

import cv2
import math
import argparse
import os

# Global state shared between the mouse callback and the main loop
_pt1     = None   # first click (object centre)
_img_disp = None  # displayed (possibly downscaled) image


def _callback_position_only(event, x, y, flags, param):
    """Single-click callback: record (x_pct, y_pct, 0) and mark the spot."""
    global _img_disp
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    h, w = _img_disp.shape[:2]
    pct_x = round(x / w, 3)
    pct_y = round(y / h, 3)
    print(f"({pct_x}, {pct_y}, 0),")
    cv2.circle(_img_disp, (x, y), 4, (0, 0, 255), -1)
    cv2.imshow("Annotate Spots", _img_disp)


def _callback_position_and_angle(event, x, y, flags, param):
    """Two-click callback: click 1 = centre, click 2 = heading direction."""
    global _pt1, _img_disp
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if _pt1 is None:
        # First click: record object centre
        _pt1 = (x, y)
        cv2.circle(_img_disp, _pt1, 4, (0, 0, 255), -1)
        cv2.imshow("Annotate Spots", _img_disp)
    else:
        # Second click: compute heading angle
        pt2 = (x, y)
        dx = pt2[0] - _pt1[0]
        dy = pt2[1] - _pt1[1]
        angle_deg = int(math.degrees(math.atan2(dy, dx)))

        cv2.line(_img_disp, _pt1, pt2, (0, 255, 0), 2)
        cv2.imshow("Annotate Spots", _img_disp)

        h, w = _img_disp.shape[:2]
        pct_x = round(_pt1[0] / w, 3)
        pct_y = round(_pt1[1] / h, 3)
        print(f"({pct_x}, {pct_y}, {angle_deg}),")

        _pt1 = None  # reset for the next spot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively annotate object placement spots on a background image."
    )
    parser.add_argument("--img",        type=str, required=True,
                        help="Path to the background image to annotate")
    parser.add_argument("--with-angle", action="store_true",
                        help="Enable two-click mode to also record heading angle")
    args = parser.parse_args()

    if not os.path.exists(args.img):
        print(f"[ERROR] Image not found: {args.img}")
        exit(1)

    _img_disp = cv2.imread(args.img)

    # Downscale very large images for display (coordinate fractions are unaffected)
    h, w = _img_disp.shape[:2]
    max_display = 1000
    if max(h, w) > max_display:
        scale = max_display / max(h, w)
        _img_disp = cv2.resize(_img_disp, (int(w * scale), int(h * scale)))

    mode = "position + angle" if args.with_angle else "position only"
    print("=" * 50)
    print(f"Image : {os.path.basename(args.img)}")
    print(f"Mode  : {mode}")
    if args.with_angle:
        print("  Click 1: object centre  (red dot)")
        print("  Click 2: heading direction  (green line)")
    else:
        print("  Click anywhere to record a placement spot.")
    print("Copy the printed tuples into BG_CONFIGS in insert_objects.py.")
    print("Press any key to quit.")
    print("=" * 50)

    callback = (_callback_position_and_angle if args.with_angle
                else _callback_position_only)

    cv2.imshow("Annotate Spots", _img_disp)
    cv2.setMouseCallback("Annotate Spots", callback)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
