# Python dependency locking

SYNTRA keeps two Python dependency files with different roles:

- `requirements.txt` is the small, human-maintained input. Update its direct version ranges when intentionally upgrading NumPy or PyTorch.
- `requirements.lock.txt` is the generated, universal pip requirements lock. It pins the full transitive dependency graph, includes SHA-256 hashes, and uses Python/platform markers to select compatible wheels.

## CPU-only PyTorch is intentional

The lock is compiled with uv's explicit CPU backend:

```powershell
uv pip compile --universal --generate-hashes --emit-index-url --torch-backend=cpu requirements.txt --output-file requirements.lock.txt
```

The lock records the CPU wheel index and selects `torch==2.15.0+cpu` on Windows/Linux. macOS uses the standard PyTorch wheel, which provides CPU/MPS runtime support rather than CUDA. The default SYNTRA environment does not install the NVIDIA CUDA runtime packages. The CPU lock is the supported default; users who need CUDA must deliberately replace the PyTorch selection with a build matching their installed CUDA stack and understand that this departs from the verified default lock.

## Regenerate and review

The generator workflow uses uv **0.13.6**, compiles the lock from `requirements.txt`, installs it with pip's `--require-hashes` mode, runs `pip check`, and verifies that `torch.version.cuda` is `None` on the Linux CPU runner. It commits the generated lock only after those checks pass.

When changing `requirements.txt`:

1. Regenerate `requirements.lock.txt` with the command above (or trigger the **Generate Python dependency lock** workflow on the feature branch).
2. Review the full lock diff. Check the PyTorch version, CPU index, platform markers, and that no `nvidia-*`, `cuda-*`, or `triton` packages have entered the default CPU lock.
3. Run the hash-locked install and validation on both Linux and Windows CI before accepting the update.

## Local validation

The native runtime installer uses the lock automatically. To validate manually in a fresh virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.lock.txt
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -c "import numpy, torch; assert torch.version.cuda is None; print(numpy.__version__, torch.__version__)"
```

Do not use `pip install -r requirements.txt` for a release or reproducibility check: it intentionally contains ranges, not the fully resolved and hashed transitive tree.
