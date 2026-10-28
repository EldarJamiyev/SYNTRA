import json
import os
import tempfile
import threading
import unittest
import torch
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path


class LocalApiSmokeTests(unittest.TestCase):
    def test_health_status_missing_checkpoint_and_origin_guard(self):
        from syntra_ai.server import Handler

        with tempfile.TemporaryDirectory(prefix="syntra-api-test-") as temp:
            missing_checkpoint = str(Path(temp) / "not-trained.pt")
            old_checkpoint = os.environ.get("SYNTRA_CHECKPOINT")
            os.environ["SYNTRA_CHECKPOINT"] = missing_checkpoint
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)

                connection.request("GET", "/health")
                response = connection.getresponse()
                health = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual(health["service"], "SYNTRA native Transformer")
                self.assertFalse(health["pretrained_weights"])

                connection.request("GET", "/model/status")
                response = connection.getresponse()
                model_status = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertFalse(model_status["checkpoint_exists"])
                self.assertFalse(model_status["external_ai_api"])

                connection.request(
                    "POST",
                    "/model/generate",
                    body=json.dumps({"prompt": "def add(a, b):"}),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                body = json.loads(response.read())
                self.assertEqual(response.status, 409)
                self.assertIn("No trained SYNTRA checkpoint", body["error"])

                from syntra_ai.config import ModelConfig
                from syntra_ai.model import SyntraTransformer

                config = ModelConfig(seq_len=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
                model = SyntraTransformer(config)
                torch.save({
                    "format": "syntra-transformer-v1",
                    "config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "step": 0,
                    "train_loss": None,
                    "validation_loss": None,
                }, missing_checkpoint)
                connection.request(
                    "POST",
                    "/model/generate",
                    body=json.dumps({
                        "prompt": "def add(",
                        "max_new_tokens": 2,
                        "temperature": 1.0,
                        "top_k": 2,
                        "device": "cpu",
                    }),
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                generated = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertTrue(generated["is_model_output"])
                self.assertGreaterEqual(generated["generated_tokens"], 1)
                self.assertLessEqual(generated["generated_tokens"], 2)

                connection.request(
                    "POST",
                    "/model/train",
                    body=json.dumps({"root": temp}),
                    headers={
                        "Content-Type": "application/json",
                        "Origin": "https://untrusted.example",
                    },
                )
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 403)
                connection.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
                if old_checkpoint is None:
                    os.environ.pop("SYNTRA_CHECKPOINT", None)
                else:
                    os.environ["SYNTRA_CHECKPOINT"] = old_checkpoint


if __name__ == "__main__":
    unittest.main()
