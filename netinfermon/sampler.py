"""sampler.py — build and find the C sampler; discover RDMA devices."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sampler", "rdma_sampler.c")
CACHE = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")),
                     "netinfermon")


def binary(build: bool = True) -> str:
    """Path of the sampler built from the current source; builds it if needed."""
    with open(SRC, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    path = os.path.join(CACHE, f"rdma_sampler-{digest}")
    if not os.path.exists(path) and build:
        cc = shutil.which("gcc") or shutil.which("cc")
        if not cc:
            raise RuntimeError("no C compiler found: install gcc (e.g. `sudo apt install gcc`)")
        os.makedirs(CACHE, exist_ok=True)
        tmp = path + ".tmp"
        subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-o", tmp, SRC], check=True)
        os.replace(tmp, path)
    return path


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def parse_rate(text: str | None) -> float | None:
    """'200 Gb/sec (4X HDR)' -> 200.0"""
    m = re.match(r"\s*([\d.]+)\s*Gb/sec", text or "")
    return float(m.group(1)) if m else None


def discover(sysfs_root: str = "/sys/class/infiniband") -> list[dict]:
    """Every RDMA device with port 1: state, link layer, port rate and counter readability."""
    out = []
    try:
        names = sorted(os.listdir(sysfs_root))
    except OSError:
        return out
    for name in names:
        port = os.path.join(sysfs_root, name, "ports", "1")
        if not os.path.isdir(port):
            continue
        counters = [os.path.join(port, "counters", c) for c in ("port_xmit_data", "port_rcv_data")]
        out.append({
            "name": name,
            "state": _read(os.path.join(port, "state")) or "?",
            "active": "ACTIVE" in (_read(os.path.join(port, "state")) or ""),
            "link_layer": _read(os.path.join(port, "link_layer")) or "?",
            "rate": _read(os.path.join(port, "rate")),
            "rate_gbps": parse_rate(_read(os.path.join(port, "rate"))),
            "counters_readable": all(_read(c) is not None for c in counters),
        })
    return out


def nominal_gbps(devices: list[dict]) -> float | None:
    """Port line rate for the active devices. Several devices on one physical port (for
    example a multi-host NIC seen through two PCIe links) each report the port's rate, so the
    highest value is used, not the sum."""
    rates = [d["rate_gbps"] for d in devices if d["active"] and d["rate_gbps"]]
    return max(rates) if rates else None
