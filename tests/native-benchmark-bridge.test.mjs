import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createNativeBenchmarkModelRequest } from "../local/native-benchmark-bridge.mjs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

test("native benchmark bridge reports only the standalone SYNTRA brain", async () => {
  const request = createNativeBenchmarkModelRequest();
  const status = await request("GET", "/model/status");
  assert.equal(status.provider, "syntra-brain-local");
  assert.equal(status.external_ai_api, false);
  assert.equal(status.local_only, true);
});

test("native benchmark bridge refuses unknown routes instead of falling back", async () => {
  const request = createNativeBenchmarkModelRequest();
  await assert.rejects(
    request("POST", "/model/provider-fallback", {}),
    /Unsupported native SYNTRA benchmark operation/
  );
});

test("benchmark command has no Ollama, Gemini, or retired model-provider dependency", async () => {
  const script = await fs.readFile(path.join(root, "scripts", "benchmark-agent.mjs"), "utf8");
  const bridge = await fs.readFile(path.join(root, "local", "native-benchmark-bridge.mjs"), "utf8");
  assert.match(script, /native-benchmark-bridge/);
  assert.doesNotMatch(script + bridge, /Ollama|Gemini|model-provider|SYNTRA_AI_PORT|SYNTRA_REVIEW_MODEL/i);
});

test("Phase 4 schedule config is bounded for a local micro-batch runtime", async () => {
  const config = JSON.parse(await fs.readFile(path.join(root, "brain", "config.json"), "utf8"));
  assert.equal(config.schema_version, 1);
  assert.equal(config.training.gradient_accumulation_steps, 4);
  assert.equal(config.training.instruction_mode, true);
  assert.equal(config.evaluation.benchmark_each_epoch, true);
  assert.deepEqual(config.evaluation.benchmark_tasks, ["utility-clamp", "python-slugify"]);
  assert.ok(config.model.block_size >= 256 && config.model.block_size <= 1024);
  assert.ok(config.training.learning_rate > 0 && config.training.learning_rate <= 0.001);
});
