#!/usr/bin/env python
"""Backup experiment outputs & upload to Google Drive via gogcli.

Adapted from https://github.com/zhugez/Neuro-Biometrics/blob/main/backup_full.py

Usage:
    python backup_full.py                         # zip weights only
    python backup_full.py --gdrive --account you@gmail.com  # + upload
    python backup_full.py --gdrive --account you@gmail.com --folder-id FOLDER_ID
"""

import zipfile
import os
import datetime
import shutil
import subprocess
import argparse

# Auto-load .env
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())


# --- ZIP WEIGHTS & OUTPUTS ---
def zip_weights():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"lvforge_backup_{timestamp}.zip"

    weight_dirs = [
        "checkpoints",
        "weights",
        "outputs",
    ]

    extra_files = [
        "README.md",
        "pyproject.toml",
    ]

    weight_exts = {".msgpack", ".pkl", ".pth", ".pt", ".json", ".html"}

    print(f"[1/3] Zipping outputs to: {zip_name}...")
    count = 0
    extra_count = 0
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zipf:
        for folder in weight_dirs:
            if not os.path.exists(folder):
                continue
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.endswith(ext) for ext in weight_exts):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, start=".")
                        zipf.write(file_path, arcname)
                        count += 1
                        print(f"  + {arcname}")

        for f in extra_files:
            if os.path.exists(f):
                zipf.write(f, f)
                extra_count += 1
                print(f"  + {f}")

    total = count + extra_count
    if total > 0:
        size_mb = os.path.getsize(zip_name) / (1024 * 1024)
        print(f"  {total} files ({count} weights, {extra_count} extra), {size_mb:.1f} MB")
        return zip_name
    else:
        if os.path.exists(zip_name):
            os.remove(zip_name)
        print("  No files found to backup.")
        return None


# --- COPY TO KAGGLE OUTPUT ---
def save_to_kaggle(filepath):
    output_dir = "/kaggle/working"
    if not os.path.isdir(output_dir):
        print(f"\n[2/3] Not a Kaggle environment. Skipping.")
        return

    dest = os.path.join(output_dir, os.path.basename(filepath))
    abs_path = os.path.abspath(filepath)

    if abs_path.startswith(output_dir):
        print(f"\n[2/3] File already in Kaggle output: {abs_path}")
    else:
        print(f"\n[2/3] Copying to Kaggle output...")
        shutil.copy2(filepath, dest)
        print(f"  -> {dest}")


# --- GOOGLE DRIVE UPLOAD via gogcli ---
GOG_KEYRING_PASSWORD = os.environ.get("GOG_KEYRING_PASSWORD", "neuro2024")


def _gog_env(account=None):
    env = {**os.environ, "GOG_KEYRING_PASSWORD": GOG_KEYRING_PASSWORD}
    if account:
        env["GOG_ACCOUNT"] = account
    return env


def _check_gog():
    try:
        r = subprocess.run(
            ["gog", "--version"], capture_output=True, text=True,
            timeout=5, env=_gog_env(),
        )
        if r.returncode == 0:
            print(f"  gogcli: {r.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("  gogcli not installed.")
    print("  Install: https://github.com/steipete/gogcli")
    return False


def _setup_gog_auth(client_secret_path, account):
    env = _gog_env(account)

    print(f"  Loading credentials from {os.path.basename(client_secret_path)}...")
    r = subprocess.run(
        ["gog", "auth", "credentials", client_secret_path],
        capture_output=True, text=True, timeout=10, env=env,
    )
    if r.returncode != 0:
        print(f"  credentials warning: {r.stderr.strip()}")

    r = subprocess.run(
        ["gog", "auth", "status"],
        capture_output=True, text=True, timeout=10, env=env,
    )
    if r.returncode == 0 and account in (r.stdout + r.stderr):
        print(f"  Already authenticated: {account}")
        return True

    print(f"\n  Authenticating {account}...")
    print("  (manual flow - copy URL to browser)\n")
    r = subprocess.run(
        ["gog", "auth", "add", account, "--services", "drive", "--manual"],
        timeout=300, env=env,
    )
    return r.returncode == 0


def upload_to_gdrive(filepath, client_secret_path, account, folder_id=None):
    print(f"\n[3/3] Google Drive Upload")

    if not _check_gog():
        return

    if not _setup_gog_auth(client_secret_path, account):
        print("  Authentication failed!")
        return

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    filename = os.path.basename(filepath)
    print(f"\n  Uploading {filename} ({file_size_mb:.1f} MB)...")

    cmd = ["gog", "drive", "upload", filepath]
    if folder_id:
        cmd.extend(["--parent", folder_id])

    env = _gog_env(account)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

    if r.returncode == 0:
        print(f"  Upload successful!")
        if r.stdout.strip():
            print(f"  {r.stdout.strip()}")
    else:
        print(f"  Upload failed: {r.stderr.strip()}")


# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup weights & upload to Google Drive")
    parser.add_argument("--gdrive", action="store_true",
                        help="Upload to Google Drive via gogcli")
    parser.add_argument("--account", type=str, default=None,
                        help="Google account email")
    parser.add_argument("--client-secret", type=str, default="client_secret.json",
                        help="Path to Google OAuth client secret JSON")
    parser.add_argument("--folder-id", type=str, default=None,
                        help="Google Drive folder ID (optional)")
    args = parser.parse_args()

    zip_file = zip_weights()
    if zip_file:
        save_to_kaggle(zip_file)

        if args.gdrive:
            if not args.account:
                print("\nError: --account required for Google Drive upload")
                print("  Example: python backup_full.py --gdrive --account you@gmail.com")
            elif not os.path.exists(args.client_secret):
                print(f"\nError: {args.client_secret} not found")
            else:
                upload_to_gdrive(zip_file, args.client_secret, args.account, args.folder_id)
        else:
            print("\nTo upload to Google Drive:")
            print("  python backup_full.py --gdrive --account you@gmail.com")

        print(f"\nDone!")
    else:
        print("No files to backup.")
