import os
from pathlib import Path

from PIL import Image  # pip install pillow

RESULTS_ROOT = r"C:\256GB\ESPCN\results"
ADDITION_ROOT = r"C:\256GB\ESPCN\addition"

TOLERANCE = 1  # allowed +/- pixels for width/height vs original


def get_image_size(path: Path):
    with Image.open(path) as img:
        return img.size  # (w, h)


def get_file_size(path: Path):
    return path.stat().st_size


def check_processed_folder(folder: Path):
    """
    Check one leaf folder under results:
    - All PNGs must have identical resolution to pass.
    - Return:
        dict with:
            'status_3' : 'PASS' / 'FAIL' for the 3-image resolution check
            'sizes'    : { (w,h): [filenames...] }
            'files'    : { 'orig': Path or None,
                           'bicubic': Path or None,
                           'espcn': Path or None }
            'dup_size' : 'NONE' / 'BICUBIC_EQ_ESPCN' / etc.
    """
    pngs = sorted(folder.glob("*.png"))  # alphabetical
    if not pngs:
        return None

    # Map specific roles if possible (orig, bicubic, espcn)
    files = {"orig": None, "bicubic": None, "espcn": None}
    for p in pngs:
        name_lower = p.name.lower()
        if "bicubic" in name_lower:
            files["bicubic"] = p
        elif "espcn" in name_lower:
            files["espcn"] = p
        else:
            # assume this is the original
            files["orig"] = p

    # resolution grouping
    sizes = {}
    for p in pngs:
        size = get_image_size(p)
        sizes.setdefault(size, []).append(p.name)

    status_3 = "PASS" if len(sizes) == 1 and len(pngs) == len(next(iter(sizes.values()))) else "FAIL"

    # duplicate check via file size (only bicubic vs espcn)
    dup_size = "NONE"
    if files["bicubic"] is not None and files["espcn"] is not None:
        size_b = get_file_size(files["bicubic"])
        size_e = get_file_size(files["espcn"])
        if size_b == size_e:
            dup_size = "BICUBIC_EQ_ESPCN"

    return {
        "folder": folder,
        "status_3": status_3,
        "sizes": sizes,
        "files": files,
        "dup_size": dup_size,
    }


def list_leaf_folders(root: Path):
    leafs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if dirnames:
            # has subfolders → not a leaf
            continue
        pngs_here = [f for f in filenames if f.lower().endswith(".png")]
        if not pngs_here:
            continue
        leafs.append(Path(dirpath))
    # sort alphabetically by folder name so 01, 02, ... align with originals
    leafs.sort(key=lambda p: p.name)
    return leafs


def list_originals_for_class(addition_root: Path, class_name: str):
    """
    Return sorted list of image Paths for the given class under addition.
    """
    folder = addition_root / class_name
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.is_file()]
    # sort alphabetically by filename
    files.sort(key=lambda p: p.name.lower())
    return files


def within_tolerance(size_proc, size_orig, tol=TOLERANCE):
    w_p, h_p = size_proc
    w_o, h_o = size_orig
    return abs(w_p - w_o) <= tol and abs(h_p - h_o) <= tol


def main():
    results_root = Path(RESULTS_ROOT)
    addition_root = Path(ADDITION_ROOT)

    classes = ["Bricks", "Fabric", "Face", "Hairs", "Nature"]

    for cls in classes:
        print(f"\n=== Class: {cls} ===")

        # all leaf folders for this class (e.g., Bricks/01, Bricks/02, ...)
        class_root = results_root / cls
        leafs = list_leaf_folders(class_root)

        # originals for this class
        originals = list_originals_for_class(addition_root, cls)

        print(f"Leaf folders (processed): {len(leafs)}")
        print(f"Original images        : {len(originals)}")

        max_pairs = min(len(leafs), len(originals))

        for i in range(max_pairs):
            folder = leafs[i]
            orig_img = originals[i]

            info = check_processed_folder(folder)
            if info is None:
                continue

            # reference size is the only size if status_3 is PASS,
            # otherwise just pick the first size as the "processed reference"
            first_size = next(iter(info["sizes"].keys()))
            size_proc = first_size
            size_orig = get_image_size(orig_img)

            status_vs_orig = "PASS" if within_tolerance(size_proc, size_orig) else "FAIL"

            print(f"\nFolder: {folder.relative_to(results_root)}")
            print(f"  Processed 3-image status: {info['status_3']}")
            for size, names in info["sizes"].items():
                print(f"    Resolution {size[0]}x{size[1]}: {len(names)} file(s)")
                for n in names:
                    print(f"      - {n}")

            print(f"  Original: {orig_img.name}")
            print(f"    Original resolution: {size_orig[0]}x{size_orig[1]}")
            print(f"    Processed ref res. : {size_proc[0]}x{size_proc[1]}")
            print(f"  Resolution vs original (±{TOLERANCE} px): {status_vs_orig}")

            # duplicate size check
            if info["dup_size"] != "NONE":
                print("  WARNING: Bicubic and ESPCN have identical file size (possible duplicate).")

            # also show file sizes if available
            b = info["files"]["bicubic"]
            e = info["files"]["espcn"]
            if b is not None:
                print(f"    Bicubic size: {get_file_size(b)} bytes")
            if e is not None:
                print(f"    ESPCN size  : {get_file_size(e)} bytes")

        # note if counts mismatch
        if len(leafs) != len(originals):
            print(
                f"\nNOTE: number of processed folders ({len(leafs)}) "
                f"and originals ({len(originals)}) differ; only first {max_pairs} pairs were compared."
            )


if __name__ == "__main__":
    main()
