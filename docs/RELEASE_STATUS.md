# Release artifact status

As of 2026-11-12, SYNTRA has no validated pretrained starter checkpoint and no published benchmark report that identifies one. The native Transformer initializes randomly and is trained locally. No checkpoint is bundled in the repository.

Do not publish a random-initialization checkpoint as a pretrained starter model. A release asset is ready only when all of the following exist together:

1. A real, completed checkpoint produced by a recorded training run.
2. Its metadata sidecar and tokenizer/inference companion, each authenticated to the exact checkpoint bytes.
3. A SHA-256 checksum and architecture/context metadata.
4. A complete benchmark report from the exact checkpoint and unchanged benchmark protocol.
5. A clean installation and startup check against the packaged artifacts.

The optional compatible-model bridge is the immediate out-of-box path for operators who already have a pretrained model endpoint. It does not turn SYNTRA's own checkpoint into pretrained weights.

No v1.0.0 tag or release is created by this change. Creating one without the above release assets would falsely suggest that a pretrained starter model and performance evidence are available.
