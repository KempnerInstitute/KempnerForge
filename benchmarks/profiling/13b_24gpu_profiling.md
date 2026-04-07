# Profiling Summary

**Date**: 2026-04-07 04:28  
**GPU**: NVIDIA H200   
**Traces**: `profiler_traces/13b_validation/`

## GPU Time Breakdown

| Category | Time (s) | % |
|----------|--------:|---:|
| MatMul/GEMM | 41.043 | 39.3 |
| Communication (NCCL) | 49.637 | 47.5 |
| Memory ops | 0.115 | 0.1 |
| Other kernels | 13.740 | 13.1 |
| **Total** | **104.535** | **100.0** |

## Efficiency

| Metric | Value |
|--------|------:|
| Total FLOPS | 22669.81 TFLOP |
| Achieved TFLOPS | 216.9 |
| GPU peak (bf16) | 989 TFLOPS |
| Kernel efficiency | 21.9% |

## Top CUDA Kernels

| Kernel | CUDA (ms) | % | Calls | GFLOPS |
|--------|----------:|---:|------:|-------:|
| record_param_comms | 24818.6 | 23.7 | 34995 | — |
| ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<... | 19265.3 | 18.4 | 11320 | — |
| nccl:_all_gather_base | 15875.2 | 15.2 | 1640 | — |
| aten::mm | 7684.6 | 7.4 | 44120 | 22668300.5 |
| nccl:all_gather_into_tensor_coalesced | 3389.8 | 3.2 | 9679 | — |
| nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT | 3239.3 | 3.1 | 7220 | — |
| ncclDevKernel_ReduceScatter_Sum_f32_RING_LL(ncclDevKernel... | 2953.4 | 2.8 | 205 | — |
| nccl:_reduce_scatter_base | 2953.4 | 2.8 | 205 | — |
| ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKerne... | 2532.3 | 2.4 | 6420 | — |
| nccl:reduce_scatter_tensor_coalesced | 2532.3 | 2.4 | 6420 | — |
| nvjet_tst_320x128_64x3_1x2_h_bz_coopB_NNT | 1934.6 | 1.9 | 5620 | — |
| aten::cat | 1760.2 | 1.7 | 19340 | — |
| aten::mul | 1473.2 | 1.4 | 61205 | 1025.1 |
| nvjet_tst_192x192_64x3_1x2_h_bz_coopB_NTN | 1149.4 | 1.1 | 1600 | — |
| void at::native::(anonymous namespace)::CatArrayBatchedCo... | 1100.6 | 1.1 | 6420 | — |
| aten::copy_ | 841.3 | 0.8 | 29425 | — |
| nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NTN | 567.0 | 0.5 | 800 | — |
| void at::native::elementwise_kernel<128, 4, at::native::g... | 565.0 | 0.5 | 19200 | — |
| aten::_cudnn_attention_backward | 521.1 | 0.5 | 800 | — |

## Viewing Traces

Load the `.json` trace files in [Perfetto UI](https://ui.perfetto.dev/) or TensorBoard:

```bash
tensorboard --logdir profiler_traces/13b_validation
```
