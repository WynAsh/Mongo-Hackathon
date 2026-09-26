"""Inputs to the planner: a workload and the hardware it must run on."""
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


class WorkloadSpec(BaseModel):
    task: str = Field(default="chat", min_length=1, max_length=2000)
    context_tokens: int = Field(default=4096, ge=128, le=1048576)
    output_tokens: int = Field(default=512, ge=1, le=131072)
    concurrency: int = Field(default=4, ge=1, le=10000)
    quality: Literal["balanced", "high"] = "balanced"
    allowed_licenses: list[str] = Field(default_factory=lambda: ["apache-2.0"])


class HardwareNode(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_.-]+$")
    cpu_model: str = "unknown"
    cpu_cores: int = Field(default=0, ge=0, le=4096)
    gpu_model: str = "unknown"
    gpu_count: int = Field(default=0, ge=0, le=256)
    vram_gb: float = Field(default=0, ge=0)
    ram_gb: float = Field(default=0, ge=0)
    storage_gb: float = Field(default=0, ge=0)
    interconnect: str = "unknown"


class HardwareInventory(BaseModel):
    os: str = "unknown"
    driver_version: str = "unknown"
    cuda_version: str = "unknown"
    kubernetes_version: str = "unknown"
    nodes: list[HardwareNode] = Field(default_factory=list, max_length=128)
    provenance: dict[str, Literal["observed", "supplied", "estimated", "unknown"]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_nodes(self):
        if len({n.name for n in self.nodes}) != len(self.nodes):
            raise ValueError("hardware node names must be unique")
        return self


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workload: WorkloadSpec = Field(default_factory=WorkloadSpec)
    inventory: HardwareInventory = Field(default_factory=HardwareInventory)
    recipe_id: Literal["kserve-llmd-vllm", "dynamo-vllm", "dynamo-sglang"] | None = None
    model_id: str | None = None
