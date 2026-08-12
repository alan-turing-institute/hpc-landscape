#!/usr/bin/env python3
"""
Fixes vllm/models/kimi_k3/nvidia/model.py
Applies the fix in vLLM commit beca88e59ea75a7aa1af72a5ae50188fa91d4e3d - "[BugFix][K3] Skip moe_intermediate padding when EP is enabled". 

Usage: fix_kimi_k3_ep_padding.py <input model.py> <output model.py>

Exits non-zero if expected broken text isn't found.
"""
import sys

BROKEN = """        if self.tp_size > 1:
            moe_intermediate_per_partition = moe_intermediate_size // self.tp_size
            if moe_intermediate_per_partition < min_moe_intermediate_per_partition:
                self.padded_moe_intermediate_size = (
                    min_moe_intermediate_per_partition * self.tp_size
                )
"""

FIXED = """        if self.tp_size > 1 and not vllm_config.parallel_config.enable_expert_parallel:
            moe_intermediate_per_partition = moe_intermediate_size // self.tp_size
            if moe_intermediate_per_partition < min_moe_intermediate_per_partition:
                self.padded_moe_intermediate_size = (
                    min_moe_intermediate_per_partition * self.tp_size
                )
"""


def main():
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <input model.py> <output model.py>")
    src_path, dst_path = sys.argv[1], sys.argv[2]

    text = open(src_path).read()
    occurrences = text.count(BROKEN)
    if occurrences != 1:
        sys.exit(
            f"Expected exactly one occurrence of the known-broken block in "
            f"{src_path}, found {occurrences}. The container's model.py has "
            f"likely changed since this patch was written -- check whether "
            f"it already includes the fix."
        )

    open(dst_path, "w").write(text.replace(BROKEN, FIXED, 1))
    print(f"Patched {src_path} -> {dst_path}")


if __name__ == "__main__":
    main()
