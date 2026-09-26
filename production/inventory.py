"""Parse supplied discovery output; never run probes on the developer machine."""
import csv
import io
import json
import re
from production.contracts import HardwareInventory

DISCOVERY_SCRIPT = '''#!/usr/bin/env bash
# Read-only inventory collection. Run on each target NVIDIA Linux server.
set -eu
printf '\\n=== HOST ===\\n'
hostname
printf '\\n=== OS ===\\n'
cat /etc/os-release
printf '\\n=== CPU ===\\n'
lscpu
printf '\\n=== GPU ===\\n'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
printf '\\n=== TOPOLOGY ===\\n'
nvidia-smi topo -m
printf '\\n=== MEMORY ===\\n'
free -b
printf '\\n=== STORAGE ===\\n'
df -B1 /var/lib
printf '\\n=== KUBERNETES ===\\n'
kubectl version -o json 2>/dev/null || true
'''


def parse_inventory(payload):
    if payload.get("format", "json") == "json":
        data = payload.get("data", {})
        if isinstance(data, str):
            data = json.loads(data)
        return HardwareInventory.model_validate(data).model_dump()
    text = payload.get("data", "")
    if not isinstance(text, str) or len(text.encode()) > 200000:
        raise ValueError("probe output must be text under 200 KB")
    sections = re.split(r"=== ([A-Z]+) ===", text)
    values = {sections[i]: sections[i+1].strip() for i in range(1, len(sections)-1, 2)}
    if not values.get("GPU"):
        raise ValueError("expected output of the supplied discovery script, including GPU section")
    gpu_rows = list(csv.reader(io.StringIO(values["GPU"])))
    groups = {}
    for row in gpu_rows:
        if len(row) != 3:
            raise ValueError("GPU rows require name, memory MiB, driver version")
        key = (row[0].strip(), float(row[1]) / 1024, row[2].strip())
        groups[key] = groups.get(key, 0) + 1
    if len(groups) != 1:
        raise ValueError("Mixed GPU types on a host require explicit JSON inventory")
    (gpu, vram, driver), count = next(iter(groups.items()))
    os_match = re.search(r'^PRETTY_NAME="?([^"\n]+)', values.get("OS", ""), re.M)
    mem_match = re.search(r"Mem:\s+(\d+)", values.get("MEMORY", ""))
    cpu_model_match = re.search(r"^Model name:\s*(.+)$", values.get("CPU", ""), re.M)
    cpu_count_match = re.search(r"^CPU\(s\):\s*(\d+)$", values.get("CPU", ""), re.M)
    storage_lines = values.get("STORAGE", "").splitlines()
    storage = 0
    if len(storage_lines) > 1:
        fields = storage_lines[-1].split()
        if len(fields) >= 4 and fields[3].isdigit():
            storage = int(fields[3]) / 1024**3
    kube = "unknown"
    try:
        kube = json.loads(values.get("KUBERNETES", "{}" )).get("serverVersion", {}).get("gitVersion", "unknown")
    except ValueError:
        pass
    data = {"os": os_match.group(1) if os_match else "unknown", "driver_version": driver,
            "cuda_version": "unknown", "kubernetes_version": kube,
            "nodes": [{"name": values.get("HOST", "target-1"),
                       "cpu_model": cpu_model_match.group(1).strip() if cpu_model_match else "unknown",
                       "cpu_cores": int(cpu_count_match.group(1)) if cpu_count_match else 0,
                       "gpu_model": gpu,
                       "gpu_count": count, "vram_gb": vram,
                       "ram_gb": int(mem_match.group(1))/1024**3 if mem_match else 0,
                       "storage_gb": storage,
                       "interconnect": "NVLink" if re.search(r"\bNV\d+\b", values.get("TOPOLOGY", "")) else "unknown"}],
            "provenance": {"os": "observed", "nodes": "observed", "driver_version": "observed",
                           "cuda_version": "unknown", "kubernetes_version": "observed" if kube != "unknown" else "unknown"}}
    return HardwareInventory.model_validate(data).model_dump()
