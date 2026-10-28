import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from syntra_ai.config import ModelConfig


@unittest.skipIf(torch is None, "Install requirements.txt to run neural model tests.")
class TransformerTests(unittest.TestCase):
    def test_random_initialized_transformer_forward_and_loss(self):
        from syntra_ai.model import SyntraTransformer
        config = ModelConfig(seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        model = SyntraTransformer(config)
        x = torch.randint(0, config.vocab_size - 1, (2, 24))
        y = torch.randint(0, config.vocab_size, (2, 24))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 24, config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)

    def test_model_generation_uses_actual_neural_forward(self):
        from syntra_ai.model import BOS_ID, SyntraTransformer
        config = ModelConfig(seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
        model = SyntraTransformer(config)
        prompt = torch.tensor([[BOS_ID, 100, 101, 102]], dtype=torch.long)
        generated = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=4)
        self.assertEqual(generated.shape[1], 9)
        self.assertEqual(generated[0, :4].tolist(), prompt[0].tolist())

    def test_config_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            ModelConfig(n_embd=190, n_head=6).validate()

    def test_real_training_checkpoint_and_supervised_prompt_masking(self):
        from syntra_ai.instruction import finetune, load_instruction_examples
        from syntra_ai.runtime import load_model
        from syntra_ai.training import train
        from syntra_ai.model import BOS_ID

        with tempfile.TemporaryDirectory(prefix="syntra-model-test-") as temp:
            root = Path(temp)
            source = root / "sample.py"
            source.write_text(
                ("def add(a, b):\n    return a + b\n\n"
                 "def clamp(value, low, high):\n    return min(max(value, low), high)\n") * 80,
                encoding="utf-8",
            )
            checkpoint = root / "checkpoints" / "model.pt"
            config = ModelConfig(seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
            result = train([str(root)], checkpoint, steps=1, batch_size=1, config=config,
                           max_tokens=10000, device_name="cpu")
            self.assertEqual(result["status"], "completed")
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(checkpoint.with_suffix(".pt.json").is_file())

            dataset = root / "instructions.jsonl"
            records = [
                {"prompt": "Write add function", "completion": "def add(a,b): return a+b"},
                {"prompt": "Write max function", "completion": "def max2(a,b): return max(a,b)"},
                {"prompt": "Write min function", "completion": "def min2(a,b): return min(a,b)"},
            ]
            dataset.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
            examples = load_instruction_examples(dataset, seq_len=32)
            first = examples[0]
            prompt_bytes = first["prompt_bytes"]
            self.assertTrue(all(token == -100 for token in first["labels"][:prompt_bytes]))
            self.assertTrue(any(token != -100 for token in first["labels"][prompt_bytes:]))

            tuned = finetune(dataset, steps=1, batch_size=1, learning_rate=1e-4,
                             device_name="cpu", checkpoint=checkpoint)
            self.assertEqual(tuned["status"], "completed")
            model, saved, device = load_model(checkpoint, "cpu")
            self.assertEqual(saved["training_mode"], "supervised-instruction-finetuning")
            self.assertEqual(saved["step"], 2)
            context = torch.tensor([[BOS_ID, *b"def add"]], dtype=torch.long, device=device)
            generated = model.generate(context, max_new_tokens=2, temperature=1.0, top_k=2)
            self.assertEqual(generated.shape[1], context.shape[1] + 2)


if __name__ == "__main__":
    unittest.main()
