#!/usr/bin/env python3
"""Fix NVFP4 EMULATION backend for compressed-tensors on GB10.

Two bugs in run_nvfp4_emulations():

1. Weight scales stored in LINEAR format but dequantize_to_dtype() calls
   convert_swizzled_to_linear() unconditionally, corrupting them.
   Root cause: convert_to_nvfp4_linear_kernel_format has no EMULATION case,
   so scales remain in their original linear format from safetensors.

2. weight_global_scale is inverted (1/actual_gs) by compressed-tensors
   process_weights_after_loading, but dequantize_to_dtype divides by it,
   effectively multiplying by actual_gs instead of dividing.

Fix: Replace run_nvfp4_emulations with a version that:
- Uses linear scales directly (no swizzle conversion)
- Detects inverted weight_global_scale (< 1.0) and re-inverts it
"""

path = "/app/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py"
with open(path) as f:
    content = f.read()

# Two upstream signatures we know about:
#   v0.19.1: positional (x, input_global_scale, weight, weight_scale_swizzled,
#            weight_global_scale); body calls dequantize_to_dtype(..., x.device,
#            group_size).
#   v0.20.0: same positional args + `swizzle: bool | None = True`; body calls
#            dequantize_to_dtype(..., output_dtype, group_size, swizzle=swizzle)
#            (drops the x.device arg, adds swizzle).
#
# For both, we replace the function body with a version that:
#   1. Re-inverts weight_global_scale if compressed-tensors stored it as 1/gs
#      (a real-world bug we hit on RedHatAI NVFP4 checkpoints).
#   2. Builds the dequantized weight directly from the linear scales, since
#      EMULATION-backend checkpoints store scales unswizzled even though the
#      tensor name still has "_swizzled".

OLD_V0_19_1 = '''def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
):
    group_size = 16
    x_m, x_k = x.shape
    output_dtype = x.dtype

    # quantize input to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = ref_nvfp4_quant(x, input_global_scale, group_size)

    # dequantize input
    x_fp4 = x_fp4.reshape(x_m, x_k // group_size, group_size)
    x_blockscale = x_blockscale.unsqueeze(-1) / input_global_scale
    x_dq = (x_fp4 * x_blockscale).reshape(x_m, x_k).to(output_dtype)
    del x_fp4, x_blockscale

    # dequantize weight
    w_fp4 = weight.data.view(torch.uint8)
    w_dq = dequantize_to_dtype(
        w_fp4,
        weight_scale_swizzled.data,
        weight_global_scale,
        output_dtype,
        x.device,
        group_size,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    del w_dq, x_dq
    return out'''

OLD_V0_20_0 = '''def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    swizzle: bool | None = True,
):
    group_size = 16
    x_m, x_k = x.shape
    output_dtype = x.dtype

    # quantize input to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = ref_nvfp4_quant(x, input_global_scale, group_size)

    # dequantize input
    x_fp4 = x_fp4.reshape(x_m, x_k // group_size, group_size)
    x_blockscale = x_blockscale.unsqueeze(-1) / input_global_scale
    x_dq = (x_fp4 * x_blockscale).reshape(x_m, x_k).to(output_dtype)
    del x_fp4, x_blockscale

    # dequantize weight
    w_fp4 = weight.data.view(torch.uint8)
    w_dq = dequantize_to_dtype(
        w_fp4,
        weight_scale_swizzled.data,
        weight_global_scale,
        output_dtype,
        group_size,
        swizzle=swizzle,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    del w_dq, x_dq
    return out'''

# Yet another body shape. Upstream v0.20.0 (post-tag-rewrite, commit 88d34c640)
# refactored the input quant/dequant into a single ref_nvfp4_quant_dequant call
# and dropped the manual `del` cleanup. Same args, same swizzle parameter.
OLD_V0_20_0_REFACTOR = '''def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    swizzle: bool | None = True,
):
    output_dtype = x.dtype
    group_size = 16

    x_dq, _ = ref_nvfp4_quant_dequant(x, input_global_scale, block_size=group_size)

    # dequantize weight
    w_fp4 = weight.data.view(torch.uint8)
    w_dq = dequantize_to_dtype(
        w_fp4,
        weight_scale_swizzled.data,
        weight_global_scale,
        output_dtype,
        group_size,
        swizzle=swizzle,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    return out'''

# vllm v0.21.0: ref_nvfp4_quant_dequant now returns a single tensor (not a
# tuple), so the unpacking on the input quant/dequant line collapsed from
# `x_dq, _ = ref_nvfp4_quant_dequant(...)` to `x_dq = ref_nvfp4_quant_dequant(...)`.
# Body is otherwise identical to the v0.20.0 refactor.
OLD_V0_21_0 = '''def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    swizzle: bool | None = True,
):
    output_dtype = x.dtype
    group_size = 16

    x_dq = ref_nvfp4_quant_dequant(x, input_global_scale, block_size=group_size)

    # dequantize weight
    w_fp4 = weight.data.view(torch.uint8)
    w_dq = dequantize_to_dtype(
        w_fp4,
        weight_scale_swizzled.data,
        weight_global_scale,
        output_dtype,
        group_size,
        swizzle=swizzle,
    )

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    return out'''

NEW_V0_20_0 = '''def run_nvfp4_emulations(
    x: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
    weight_global_scale: torch.Tensor,
    swizzle: bool | None = True,
):
    group_size = 16
    x_m, x_k = x.shape
    output_dtype = x.dtype

    # quantize input to (FP4 and interleaved block scale)
    x_fp4, x_blockscale = ref_nvfp4_quant(x, input_global_scale, group_size)

    # dequantize input
    x_fp4 = x_fp4.reshape(x_m, x_k // group_size, group_size)
    x_blockscale = x_blockscale.unsqueeze(-1) / input_global_scale
    x_dq = (x_fp4 * x_blockscale).reshape(x_m, x_k).to(output_dtype)
    del x_fp4, x_blockscale

    # dequantize weight - FIXED for EMULATION backend on GB10
    # Bug 1: Weight scales are LINEAR (not swizzled) in EMULATION mode
    #   convert_to_nvfp4_linear_kernel_format has no EMULATION case
    # Bug 2: weight_global_scale is inverted (1/actual_gs) by compressed-tensors
    #   but dequantize formula needs: w_fp4 * block_scale / actual_gs
    wgs = weight_global_scale
    if wgs.item() < 1.0:
        wgs = 1.0 / wgs

    w_fp4 = weight.data.view(torch.uint8)
    m, packed_k = w_fp4.shape
    k = packed_k * 2
    tensor_f32 = break_fp4_bytes(w_fp4, torch.float32)
    tensor_f32 = tensor_f32.reshape(m, k // group_size, group_size)
    w_sf = weight_scale_swizzled.data.view(torch.float8_e4m3fn)
    w_sf_linear = w_sf[:m, :k // group_size]
    w_sf_dtype = w_sf_linear.to(torch.float32) / wgs
    w_dq = (tensor_f32 * w_sf_dtype.unsqueeze(-1)).reshape(m, k).to(output_dtype)

    # matmul
    out = torch.matmul(x_dq, w_dq.t())
    del w_dq, x_dq
    return out'''

# Sentinel kept by the patch so we can detect "already-patched" idempotently.
PATCH_SENTINEL = "# Bug 1: Weight scales are LINEAR (not swizzled) in EMULATION mode"

if PATCH_SENTINEL in content:
    print("Already patched (sentinel present)")
elif OLD_V0_21_0 in content:
    content = content.replace(OLD_V0_21_0, NEW_V0_20_0)
    with open(path, 'w') as f:
        f.write(content)
    print("Fix applied to v0.21.0 signature: EMULATION backend weight dequantization")
elif OLD_V0_20_0_REFACTOR in content:
    content = content.replace(OLD_V0_20_0_REFACTOR, NEW_V0_20_0)
    with open(path, 'w') as f:
        f.write(content)
    print("Fix applied to v0.20.0 (refactored) signature: EMULATION backend weight dequantization")
elif OLD_V0_20_0 in content:
    content = content.replace(OLD_V0_20_0, NEW_V0_20_0)
    with open(path, 'w') as f:
        f.write(content)
    print("Fix applied to v0.20.0 signature: EMULATION backend weight dequantization")
elif OLD_V0_19_1 in content:
    # Use the old NEW string (without swizzle) for v0.19.x
    NEW_V0_19_1 = NEW_V0_20_0.replace(
        "    weight_global_scale: torch.Tensor,\n    swizzle: bool | None = True,\n",
        "    weight_global_scale: torch.Tensor,\n",
    )
    content = content.replace(OLD_V0_19_1, NEW_V0_19_1)
    with open(path, 'w') as f:
        f.write(content)
    print("Fix applied to v0.19.x signature: EMULATION backend weight dequantization")
else:
    print("ERROR: Could not find run_nvfp4_emulations in any known form (v0.19.1, v0.20.0, v0.20.0-refactor, v0.21.0)")
    idx = content.find('def run_nvfp4_emulations')
    if idx >= 0:
        print("Current code starts with:")
        print(content[idx:idx + 600])
    exit(1)
