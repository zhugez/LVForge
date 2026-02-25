#!/bin/bash
export PATH="/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export HOME="/root"
cd /mnt/e/Test/Test/LVForge
uv pip install -e ".[dev]"
