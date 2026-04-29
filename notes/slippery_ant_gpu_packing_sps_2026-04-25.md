# SlipperyAnt GPU Packing vs SPS - 2026-04-25

## Observation

Single default run on one H100:

- About `16,000 SPS`

Eight concurrent default runs on one H100:

- About `2,000 SPS` per run
- Aggregate: `8 * 2,000 = 16,000 SPS`

## Interpretation

Eight-way packing fits in memory, but it does not increase total throughput.

The original high memory usage was mostly JAX allocator reservation. After lowering `XLA_PYTHON_CLIENT_MEM_FRACTION`, many processes can fit on one H100. However, once packed, the processes compete for the same GPU execution resources. Each run slows down roughly in proportion to the number of concurrent runs, leaving aggregate SPS approximately flat.

Practical conclusion:

- Memory is not the bottleneck for state-observation SlipperyAnt PPO.
- GPU compute/kernel scheduling is the bottleneck.
- Multi-process packing is useful for convenience or reducing SLURM job count.
- Multi-process packing is not useful for increasing aggregate environment steps per GPU.

## Recommendation

Use one run per GPU when fast per-run turnaround matters.

Use process packing only when it is acceptable for each run to progress more slowly, for example for low-priority sweeps where convenience matters more than wall-clock time per config.

To increase aggregate SPS per GPU, prefer a single process with more vectorized work, such as larger `N_ENVS` and/or `NUM_STEPS`, rather than many independent Python/JAX processes on the same GPU.
