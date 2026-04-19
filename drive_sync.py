"""
drive_sync.py
=============
Utilities for saving and restoring DEHB checkpoints + log data
to/from Google Drive in Colab.

Usage
─────
  # At the end of a training session — save everything to Drive:
  from drive_sync import save_to_drive
  save_to_drive()

  # At the start of a new session — restore from Drive:
  from drive_sync import load_from_drive
  load_from_drive()
"""

import os
import shutil
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DRIVE_MOUNT    = "/content/drive"
DRIVE_SAVE_DIR = f"{DRIVE_MOUNT}/MyDrive/snake_hpo"

LOCAL_DIRS  = ["dehb_checkpoints", "dehb_log_data"]
LOCAL_FILES = ["best_snake_params.json"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mount_drive() -> bool:
    """Mount Google Drive if not already mounted. Returns True on success."""
    if Path(f"{DRIVE_MOUNT}/MyDrive").exists():
        print("[Drive] Already mounted")
        return True
    try:
        from google.colab import drive
        drive.mount(DRIVE_MOUNT)
        return True
    except Exception as e:
        print(f"[Drive] Mount failed: {e}")
        return False


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _print_drive_usage(drive_dir: str) -> None:
    total = 0
    print("\n[Drive] Saved files:")
    for p in sorted(Path(drive_dir).iterdir()):
        size = p.stat().st_size if p.is_file() else _dir_size(p)
        total += size
        print(f"  {p.name:40} {size / 1e6:8.1f} MB")
    print(f"  {'TOTAL':40} {total / 1e6:8.1f} MB")


def _print_local_summary() -> None:
    print("\n[Local] Restored:")
    for dirname in LOCAL_DIRS:
        if Path(dirname).exists():
            size = _dir_size(Path(dirname))
            n    = sum(1 for _ in Path(dirname).rglob("*") if _.is_file())
            print(f"  {dirname:30} {size / 1e6:8.1f} MB  ({n} files)")
    for fname in LOCAL_FILES:
        if Path(fname).exists():
            size = Path(fname).stat().st_size
            print(f"  {fname:30} {size / 1e3:8.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save_to_drive(
    drive_dir: str = DRIVE_SAVE_DIR,
    compress:  bool = True,
) -> None:
    """
    Save DEHB checkpoints, log data, and best params to Google Drive.

    Parameters
    ----------
    drive_dir : str
        Destination folder on Drive. Created if it doesn't exist.
    compress : bool
        If True, zip directories before uploading (faster transfer,
        less Drive quota). If False, copy files directly — slower but
        no memory overhead and nothing to go wrong with zipping.
    """
    if not _mount_drive():
        return

    os.makedirs(drive_dir, exist_ok=True)
    print(f"[Drive] Saving to {drive_dir} ...")

    # ── Directories ────────────────────────────────────────────────
    for dirname in LOCAL_DIRS:
        if not Path(dirname).exists():
            print(f"  [skip] {dirname}/ not found locally")
            continue

        if compress:
            zip_path = f"{dirname}.zip"
            print(f"  Compressing {dirname}/ → {zip_path} ...")

            try:
                shutil.make_archive(dirname, "zip", dirname)
            except Exception as e:
                print(f"  [ERROR] Failed to zip {dirname}: {e}")
                print(f"  Retrying as direct copy...")
                try:
                    dest = f"{drive_dir}/{dirname}"
                    if Path(dest).exists():
                        shutil.rmtree(dest)
                    shutil.copytree(dirname, dest)
                    print(f"  ✓ {dest}/  (direct copy)")
                except Exception as e2:
                    print(f"  [ERROR] Direct copy also failed: {e2}")
                continue

            # Verify zip is non-empty before uploading
            if not Path(zip_path).exists() or Path(zip_path).stat().st_size == 0:
                print(f"  [ERROR] Zip file missing or empty: {zip_path}")
                continue

            zip_size_mb = Path(zip_path).stat().st_size / 1e6
            print(f"  Zip size: {zip_size_mb:.1f} MB — uploading ...")

            dest = f"{drive_dir}/{zip_path}"
            try:
                shutil.copy(zip_path, dest)
            except Exception as e:
                print(f"  [ERROR] Upload failed: {e}")
                print(f"  Local zip kept at {zip_path} — copy manually")
                continue

            # Verify Drive copy matches before deleting local zip
            if (Path(dest).exists() and
                    Path(dest).stat().st_size == Path(zip_path).stat().st_size):
                os.remove(zip_path)
                print(f"  ✓ {dest}  ({zip_size_mb:.1f} MB)")
            else:
                print(f"  [WARN] Drive copy size mismatch — keeping local zip at {zip_path}")

        else:
            # Direct copy — no zip, no memory overhead
            dest = f"{drive_dir}/{dirname}"
            print(f"  Copying {dirname}/ → {dest}/ ...")
            try:
                if Path(dest).exists():
                    shutil.rmtree(dest)
                shutil.copytree(dirname, dest)
                print(f"  ✓ {dest}/")
            except Exception as e:
                print(f"  [ERROR] Failed to copy {dirname}: {e}")

    # ── Loose files ────────────────────────────────────────────────
    for fname in LOCAL_FILES:
        if not Path(fname).exists():
            print(f"  [skip] {fname} not found locally")
            continue
        dest = f"{drive_dir}/{fname}"
        try:
            shutil.copy(fname, dest)
            print(f"  ✓ {dest}")
        except Exception as e:
            print(f"  [ERROR] Failed to copy {fname}: {e}")

    print("\n[Drive] Save complete.")
    print("  Tip: if any item failed, retry with save_to_drive(compress=False)")
    _print_drive_usage(drive_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_from_drive(
    drive_dir: str = DRIVE_SAVE_DIR,
    overwrite: bool = False,
) -> None:
    """
    Restore DEHB checkpoints and log data from Google Drive.

    Parameters
    ----------
    drive_dir : str
        Source folder on Drive.
    overwrite : bool
        If False (default), skip items that already exist locally — safe
        for resuming a session without clobbering anything.
        If True, always overwrite local copies.
    """
    if not _mount_drive():
        return

    if not Path(drive_dir).exists():
        print(f"[Drive] Source directory not found: {drive_dir}")
        print("  Have you run save_to_drive() in a previous session?")
        return

    print(f"[Drive] Restoring from {drive_dir} ...")

    # ── Directories (try zip first, fall back to folder) ───────────
    for dirname in LOCAL_DIRS:
        zip_src = f"{drive_dir}/{dirname}.zip"
        dir_src = f"{drive_dir}/{dirname}"

        if Path(zip_src).exists():
            if Path(dirname).exists() and not overwrite:
                print(f"  [skip] {dirname}/ already exists locally (overwrite=False)")
                continue
            print(f"  Extracting {dirname}.zip ...")
            try:
                shutil.unpack_archive(zip_src, dirname)
                print(f"  ✓ {dirname}/")
            except Exception as e:
                print(f"  [ERROR] Failed to extract {zip_src}: {e}")

        elif Path(dir_src).exists():
            if Path(dirname).exists() and not overwrite:
                print(f"  [skip] {dirname}/ already exists locally (overwrite=False)")
                continue
            print(f"  Copying {dirname}/ ...")
            try:
                if Path(dirname).exists():
                    shutil.rmtree(dirname)
                shutil.copytree(dir_src, dirname)
                print(f"  ✓ {dirname}/")
            except Exception as e:
                print(f"  [ERROR] Failed to copy {dir_src}: {e}")

        else:
            print(f"  [skip] {dirname} not found on Drive")

    # ── Loose files ────────────────────────────────────────────────
    for fname in LOCAL_FILES:
        src = f"{drive_dir}/{fname}"
        if not Path(src).exists():
            print(f"  [skip] {fname} not found on Drive")
            continue
        if Path(fname).exists() and not overwrite:
            print(f"  [skip] {fname} already exists locally (overwrite=False)")
            continue
        try:
            shutil.copy(src, fname)
            print(f"  ✓ {fname}")
        except Exception as e:
            print(f"  [ERROR] Failed to copy {fname}: {e}")

    print("\n[Drive] Restore complete.")
    _print_local_summary()
