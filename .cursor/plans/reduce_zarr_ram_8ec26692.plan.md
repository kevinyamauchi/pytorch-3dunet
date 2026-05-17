---
name: Reduce Zarr RAM
overview: Profile the StimCycle follicle training job on the local debug bundle, then reduce avoidable host RAM in the training data path. The plan explicitly tests whether memory scales with 1/5/20/100 files and whether smaller batch sizes or worker counts reduce RAM without excessive slowdown.
todos:
  - id: branch-and-configs
    content: Create the memory-fix branch and local short-run subset configs after approval.
    status: completed
  - id: memory-monitor
    content: Add or create a lightweight RSS process-tree monitor for reproducible training measurements.
    status: completed
  - id: baseline-matrix
    content: Run baseline training on 1, 5, 20, and 100 zarr files.
    status: completed
  - id: batch-worker-sweep
    content: Measure batch size and DataLoader worker count tradeoffs for RAM and training throughput.
    status: completed
  - id: train-memory-fixes
    content: Address any training-time materialization or avoidable float64/worker memory found by the baseline matrix.
    status: completed
  - id: tests-and-rerun
    content: Add focused lazy zarr tests and re-run the same memory matrix to verify improvement.
    status: completed
isProject: false
---

# Plan To Reduce StimCycle Training RAM

## Current Read-Only Findings

- The active repo is `[pytorch-3dunet](/local0/home/lfranz/code/pytorch-3dunet)` on `memory_dataset`, latest commit `565e1b9`.
- The copied debug inputs live in `[/local1/lfranz/local_data/debug_training_memory](/local1/lfranz/local_data/debug_training_memory)` with `follicle_3dunet_training_config.yaml` and `combined_zarr/`.
- `LazyZarrDataset` is now present at `[/local0/home/lfranz/code/pytorch-3dunet/pytorch3dunet/datasets/zarr_dataset.py](/local0/home/lfranz/code/pytorch-3dunet/pytorch3dunet/datasets/zarr_dataset.py)`.
- Training does not use test-time `mirror_padding`, so the first focus is the train data path: `batch_size: 15`, `num_workers: 8`, float64 zarr input, patch transforms, PyTorch prefetching, and `FilterSliceBuilder` scanning label patches over every zarr file.
- The key training question is whether RAM grows with the number of files. If it does, lazy loading is likely being defeated during dataset construction or slice filtering. If it does not, the dominant cost is probably fixed per-run pipeline memory: workers, prefetched batches, patch transform temporaries, model staging, and batch size.

## Execution Plan After Approval

1. Create an isolated branch from the updated repo, e.g. `fix/stimcycle-zarr-memory`, and keep generated debug configs/logs under the local debug folder or an ignored diagnostics folder.
2. Build a lightweight process-tree memory monitor script that launches training commands and records timestamped RSS/peak RSS for the parent plus DataLoader worker children. Use this for every run so results are comparable.
3. Generate local, capped copies of the StimCycle configs:
  - Replace `/cluster/project/.../StimCycle/combined_zarr` with `[/local1/lfranz/local_data/debug_training_memory/combined_zarr](/local1/lfranz/local_data/debug_training_memory/combined_zarr)`.
  - Replace output/checkpoint paths with local scratch paths.
  - Cap training to a short run that gets through dataset construction and several batches.
  - Create train subsets with 1, 5, 20, and 100 existing zarr stores; keep validation tiny unless validation memory is under test.
4. Run the baseline training matrix with current code:
  - 1, 5, 20, 100 file subsets.
  - Record peak RSS, steady-state RSS after first batches, time to first batch, and whether memory rises during dataset creation or during iteration.
  - Interpret scaling: strong growth with file count means lazy loading is being defeated; flat memory means fixed per-run costs dominate.
5. Run a batch-size and worker-count sweep on a representative subset:
  - Batch sizes: start with `15` baseline, then test smaller values such as `10`, `5`, `2`, and `1` if needed.
  - Worker counts: test `8` baseline, then `4`, `2`, `1`, and `0`.
  - Measure both peak RSS and useful throughput, such as seconds per training iteration after warmup and time to first batch.
  - Prefer configurations that sharply reduce RAM with modest throughput loss over changes that only shift memory into slower I/O.
6. Implement training fixes in small iterations, measuring after each:
  - Downcast raw patches from float64 to float32 before tensor conversion if the model trains in float32 and no transform already does this.
  - Keep `LazyZarrDataset.ds_stats()` from full reads; add tests/guards so future stats or filtering changes cannot call `np.asarray`, `ravel`, or full-volume `np.pad` on lazy zarr arrays.
  - Inspect and optimize `FilterSliceBuilder` if file-count scaling shows it reads or copies too much label data during dataset construction.
  - Tune DataLoader behavior if worker/prefetch memory dominates, potentially by exposing `prefetch_factor`, `persistent_workers`, and `pin_memory` config controls only if the measurements justify it.
7. Re-run the same 1/5/20/100 training matrix and the best batch/worker candidates after each fix. Keep the change only if peak RSS drops without breaking patch shapes or training batches.
8. Add focused tests around `LazyZarrDataset`:
  - Training slice filtering reads label patches only, not full raw volumes.
  - Dataset construction with lazy zarr arrays does not call whole-array materialization.
  - Float32 patch conversion, if added, preserves expected tensor shapes and label behavior.
9. Report the final before/after memory curves, throughput tradeoffs, remaining irreducible memory costs, and recommended production config changes such as lower `batch_size`, tuned `num_workers`, DataLoader prefetch settings, and float32 zarr storage if needed.

## Expected Decision Points

- If memory grows roughly linearly with file count during dataset construction, prioritize loader/slice-builder materialization bugs.
- If memory is flat across file counts but high during first batches, prioritize batch size, workers, prefetching, transform temporaries, and float64-to-float32 patch conversion.
- If lower worker counts reduce RAM a lot but slow training too much, tune prefetching and batch size before accepting a very low-worker configuration.

