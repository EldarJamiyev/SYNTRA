# Data and laptop resource guidance

## Local corpus

SYNTRA reads the absolute local directory selected in the Project Builder or passed to `python -m syntra_ai.cli train --root ...`. It does not silently fetch source code or training sets. Supported text/code files are read as UTF-8. Common dependency/build/cache folders, symlinks, hidden directories, likely secret/credential files, non-text formats and files larger than 256 KiB are skipped. The default total token cap is two million in the CLI and the UI; the Python API can be configured with a higher limit.

Do not train on confidential work files, secrets, private repositories, or material whose license does not permit the intended use. The exclusion rules are only heuristics and cannot guarantee that every secret will be recognized. Review the corpus and permissions before training. The repository does not bundle a coding dataset or model checkpoint.

## Default architecture and resources

The native model default is a small decoder-only Transformer: 258 byte/BOS/EOS vocabulary IDs, a 1,024-byte context, four layers, four attention heads, width 128 and batch size four in the UI. It has roughly one million parameters. Actual memory and throughput depend on OS, PyTorch build, selected device, sequence length, batch size and other running programs.

Use a smaller configuration for a first smoke run if it is slow:
- `--seq-len 128`
- `--layers 1`
- `--heads 2`
- `--width 32`
- `--batch-size 1`
- `--steps 1` or `10`

These settings verify the end-to-end training/checkpoint pipeline only. They are not expected to produce useful coding abilities. Normal byte-level training needs a substantial corpus; the 80/20 split must leave enough byte tokens on both sides of the split for the configured context.

Example:
```powershell
python -m syntra_ai.cli inspect-data --root "C:\Users\You\Documents\YourProject"
python -m syntra_ai.cli train --root "C:\Users\You\Documents\YourProject" --steps 1000
python -m syntra_ai.cli status
```

## Pagefile / virtual memory

Windows-managed virtual memory can reduce the chance that a process immediately fails when committed memory rises, but pagefile-backed memory is much slower than RAM. It does not increase CPU/GPU throughput. Prefer closing memory-heavy applications and reducing batch size, context length, model width or layer count before relying on a larger pagefile. SYNTRA does not change Windows pagefile settings.

## Hardware acceleration

The default requirements install CPU-only PyTorch to avoid pulling multi-gigabyte NVIDIA runtime packages onto laptops without NVIDIA GPUs. The model code can choose CUDA or Apple MPS when PyTorch is installed with a matching GPU build. Intel Arc GPU acceleration through DirectML or XPU is not currently implemented; an integrated Intel Arc GPU on a Windows laptop is therefore not used by the default installation.

## Metrics and quality

The checkpoint metadata records measured training and held-out validation loss. It is a record of next-byte prediction, not a score for solving software tasks. Compare separate coding tasks and measure syntax validity, patch acceptance, build/test pass rate and regressions. Keep an independent hold-out set when tuning instruction data. Never treat a lower loss or the existence of a checkpoint as proof that generated code is correct.
