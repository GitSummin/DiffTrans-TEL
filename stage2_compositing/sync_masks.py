"""
Stage 2a — Mask / Object File Synchronisation

After manually reviewing extracted masks, unwanted samples are removed by
deleting their mask file.  This script removes the corresponding object (RGBA)
file so that the objects/ and masks/ directories remain in sync.

Usage:
  python sync_masks.py --dir extracted_results/<class_name>/
"""

import argparse
from pathlib import Path


def sync_folders(target_dir):
    """Delete object files whose paired mask file no longer exists."""
    target_path = Path(target_dir)
    masks_dir   = target_path / "masks"
    objects_dir = target_path / "objects"

    if not masks_dir.exists() or not objects_dir.exists():
        print("[ERROR] 'masks' or 'objects' subdirectory not found under:", target_dir)
        return

    # Use pathlib.glob to safely handle directory names containing brackets
    obj_files = list(objects_dir.glob("*_obj.png"))

    deleted_count = 0
    for obj_path in obj_files:
        base_name = obj_path.name.replace("_obj.png", "")
        expected_mask = masks_dir / f"{base_name}_mask.png"

        if not expected_mask.exists():
            obj_path.unlink()
            print(f"[REMOVED] {obj_path.name}  (no matching mask)")
            deleted_count += 1

    print(f"\n[DONE] {deleted_count} orphaned object file(s) removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove object files whose corresponding mask has been deleted."
    )
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory containing 'masks/' and 'objects/' subdirectories")
    args = parser.parse_args()
    sync_folders(args.dir)
