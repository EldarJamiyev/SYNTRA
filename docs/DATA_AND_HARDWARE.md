# Data and laptop resource guidance

## Corpus preparation

Put code/text files you have the right to use in data/raw. The tool reads supported local text file extensions, ignores common dependency/build folders, skips files larger than 2 MB, and enforces a default 50 MB source-byte budget. These bounds protect an initial laptop workflow; they are not recommended final training-corpus sizes for a strong coding model.

Do not train on confidential work files, secrets, private repositories, or material whose license does not permit the intended use. Keep raw and processed data out of Git.

## Expected memory behavior

The default configuration is a small approximately 7–8 million parameter transformer, 128-token context, and batch size 4. Actual memory use depends on PyTorch version, tensor allocation, Windows, browser tabs, and other applications. Start with the default settings and reduce n_embd, n_layer, block_size, and batch_size if training is unstable or too slow.

Suggested first-run settings:

- n_embd: 192
- n_layer: 2
- n_head: 4
- block_size: 64
- batch_size: 2
- max_steps: 50

Use these tiny settings only to check the pipeline. They are not expected to produce a useful code model.

## Pagefile / virtual memory

Windows-managed virtual memory can reduce the chance that a process immediately fails when committed memory rises, but pagefile-backed memory is much slower than RAM. It does not increase CPU/GPU speed, and heavy paging can make training impractically slow. Prefer closing memory-heavy applications and reducing the batch/context/model dimensions before relying on a large pagefile.

The repository intentionally does not change Windows pagefile settings automatically. Configure Windows virtual memory manually only if you understand the disk-space and system-performance trade-offs.

## Hardware acceleration

The initial runtime automatically selects CUDA when PyTorch reports it is available; otherwise it uses CPU. An integrated Intel Arc GPU is not a CUDA device and is not used by this CUDA-only selection logic. This implementation runs without a discrete GPU, with slower training throughput.

## Measuring progress

Monitor checkpoints/training_metrics.csv and compare training versus validation loss. Falling training loss with rising validation loss is a common overfitting sign. Always retain held-out code tasks and run syntax/unit tests against generated completions before treating improvements as real.
