# Phase 22 — Preserve model mode across inference helpers

## Defect

The public native inference methods \`SyntraTransformer.generate()\` and \`SyntraTransformer.embed_tokens()\` called \`self.eval()\` but did not restore the model's previous mode. Calling either method on a model still used for training silently left dropout and other training-sensitive modules disabled for subsequent optimization steps. A failing inference call had the same side effect because the mode switch happened before the forward operation.

A root-level \`model.train(previous_flag)\` restoration is insufficient for a module tree that deliberately uses mixed modes. PyTorch modules can have different training flags at different depths; recursively setting the root flag would erase those intentional settings.

## Implementation

- Add a small inference-call decorator that snapshots the \`training\` flag of every module in the model tree.
- Restore each module's exact flag in a \`finally\` block, regardless of whether inference returns successfully or raises.
- Apply the decorator to both \`generate()\` and \`embed_tokens()\`.
- Retain the existing \`torch.no_grad()\` behavior and evaluation-mode forward execution.
- Keep this behavior inside the native model; no runtime/provider architecture changes are required.

## Regression coverage

Two native model tests cover:
1. Successful embedding and generation calls restore every module's original flag, including an intentionally mixed attention/dropout configuration.
2. A forced forward failure in each helper still restores the complete module-mode snapshot.

The tests compare all module training flags before and after each call, rather than checking only the root model's \`training\` value.

## Boundaries

This is an inference lifecycle correctness fix. It does not modify model parameters, checkpoints, optimizer state, data, sampling behavior, or model quality. It does not claim any improvement in reasoning or coding accuracy.
