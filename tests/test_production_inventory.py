import pytest
from production.inventory import parse_inventory, DISCOVERY_SCRIPT


def test_probe_import_reports_observed_vs_unknown():
    inventory = parse_inventory({"format": "probe", "data": '''=== HOST ===
gpu-a
=== OS ===
PRETTY_NAME="Ubuntu 24.04"
=== CPU ===
CPU(s):                          64
Model name:                      AMD EPYC 9554P 64-Core Processor
=== GPU ===
NVIDIA H100, 81920, 580.95.05
NVIDIA H100, 81920, 580.95.05
=== TOPOLOGY ===
GPU0 GPU1 NV18
=== MEMORY ===
Mem: 274877906944 1 2 3
=== STORAGE ===
Filesystem 1B-blocks Used Available Use% Mounted on
/dev/sda 1099511627776 1 549755813888 50% /var/lib
=== KUBERNETES ===
{"serverVersion":{"gitVersion":"v1.34.0"}}
'''})
    assert inventory["nodes"][0]["gpu_count"] == 2
    assert inventory["nodes"][0]["cpu_cores"] == 64
    assert inventory["nodes"][0]["cpu_model"].startswith("AMD EPYC")
    assert inventory["nodes"][0]["vram_gb"] == 80
    assert inventory["nodes"][0]["storage_gb"] == 512
    assert inventory["nodes"][0]["interconnect"] == "NVLink"
    assert inventory["provenance"]["cuda_version"] == "unknown"
    assert inventory["kubernetes_version"] == "v1.34.0"


def test_bad_probe_or_duplicate_nodes_rejected():
    with pytest.raises(ValueError):
        parse_inventory({"format": "probe", "data": "random text"})
    with pytest.raises(ValueError):
        parse_inventory({"data": {"nodes": [{"name": "same"}, {"name": "same"}]}})


def test_discovery_has_no_install_commands():
    assert "nvidia-smi" in DISCOVERY_SCRIPT
    assert "kubectl apply" not in DISCOVERY_SCRIPT
    assert "apt install" not in DISCOVERY_SCRIPT
