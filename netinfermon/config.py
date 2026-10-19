"""config.py — settings, from ~/.config/netinfermon/config.toml and NETINFERMON_* environment variables.

Example config.toml (every key is optional):

    endpoint = "http://127.0.0.1:8000/v1"   # default model endpoint for benchmarks
    web_host = "0.0.0.0"
    web_port = 8502
    api_port = 8599                         # local control API, 127.0.0.1 only
    interval_us = 1000                      # sampling interval
    devices = []                            # RDMA devices to read; empty = every ACTIVE one
    insecure_tls = false                    # true: accept self-signed https endpoints

    [link]
    max_gbps = 195          # measured maximum with all devices (per direction)
    one_device_gbps = 109   # measured maximum with traffic on one device only

Environment variables override the file: NETINFERMON_DATA, NETINFERMON_ENDPOINT, NETINFERMON_WEB_PORT,
NETINFERMON_API_PORT, NETINFERMON_INTERVAL_US, NETINFERMON_SYSFS (sysfs root, for tests),
NETINFERMON_LINK_MAX_GBPS, NETINFERMON_ONE_DEVICE_GBPS, NETINFERMON_INSECURE_TLS=1.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field

CONFIG_PATH = os.environ.get("NETINFERMON_CONFIG",
                             os.path.join(os.path.expanduser("~"), ".config", "netinfermon", "config.toml"))


@dataclass
class Settings:
    data_dir: str
    endpoint: str = "http://127.0.0.1:8000/v1"
    web_host: str = "0.0.0.0"
    web_port: int = 8502
    api_port: int = 8599
    interval_us: int = 1000
    sysfs_root: str = "/sys/class/infiniband"
    devices: list[str] = field(default_factory=list)
    link_max_gbps: float | None = None
    one_device_gbps: float | None = None
    insecure_tls: bool = False

    @property
    def api_url(self) -> str:
        return f"http://127.0.0.1:{self.api_port}"

    def run_dir(self, campaign: str, label: str) -> str:
        return os.path.join(self.data_dir, campaign, label)

    @property
    def recordings_dir(self) -> str:
        return os.path.join(self.data_dir, "_recordings")


def load() -> Settings:
    raw: dict = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
    link = raw.get("link", {})
    env = os.environ.get
    s = Settings(
        data_dir=env("NETINFERMON_DATA") or raw.get("data_dir")
        or os.path.join(os.path.expanduser("~"), ".local", "share", "netinfermon"),
        endpoint=env("NETINFERMON_ENDPOINT") or raw.get("endpoint", Settings.endpoint),
        web_host=raw.get("web_host", Settings.web_host),
        web_port=int(env("NETINFERMON_WEB_PORT") or raw.get("web_port", Settings.web_port)),
        api_port=int(env("NETINFERMON_API_PORT") or raw.get("api_port", Settings.api_port)),
        interval_us=int(env("NETINFERMON_INTERVAL_US") or raw.get("interval_us", Settings.interval_us)),
        sysfs_root=env("NETINFERMON_SYSFS") or raw.get("sysfs_root", Settings.sysfs_root),
        devices=list(raw.get("devices", [])),
        link_max_gbps=_float(env("NETINFERMON_LINK_MAX_GBPS") or link.get("max_gbps")),
        one_device_gbps=_float(env("NETINFERMON_ONE_DEVICE_GBPS") or link.get("one_device_gbps")),
        insecure_tls=env("NETINFERMON_INSECURE_TLS") == "1" or bool(raw.get("insecure_tls", False)),
    )
    s.data_dir = os.path.expanduser(s.data_dir)
    return s


def _float(v) -> float | None:
    return None if v in (None, "") else float(v)
