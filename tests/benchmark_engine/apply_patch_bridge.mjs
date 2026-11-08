import fs from "node:fs/promises";
import { validatePatch, writeCandidateChanges } from "../../local/agent.mjs";

async function main() {
  const workspaceInput = process.argv[2];
  const payloadPath = process.argv[3];
  if (!workspaceInput || !payloadPath) throw new Error("Usage: apply_patch_bridge.mjs <workspace> <payload.json>");
  const workspace = await fs.realpath(workspaceInput);
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  if (!payload || typeof payload !== "object" || !payload.patch || !Array.isArray(payload.expected_paths)) {
    throw new Error("Benchmark patch bridge received an invalid payload.");
  }
  if (payload.patch.format !== "SYNTRA_AGENT_PATCH_V1") throw new Error("Unsupported structured patch format.");
  const validated = await validatePatch(workspace, payload.patch);
  if (validated.changes.some(change => change.existedBefore !== true)) {
    throw new Error("Raw-model benchmark patches may not create files.");
  }
  const actual = validated.changes.map(change => change.path).sort();
  const expected = [...payload.expected_paths].sort();
  if (actual.length !== expected.length || actual.some((item, index) => item !== expected[index])) {
    throw new Error("Structured patch changed-path set does not match the fixture's expected files.");
  }
  await writeCandidateChanges(workspace, validated.changes);
  process.stdout.write(JSON.stringify({ changed_paths: actual }));
}

main().catch(error => {
  process.stderr.write((error instanceof Error ? error.message : String(error)) + "\n");
  process.exitCode = 1;
});
