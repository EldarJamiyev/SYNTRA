const VALID_PATCH_ATTEMPT_STATES = new Set([
  "patch_validated",
  "verification_passed",
  "verification_passed_review_required",
  "verification_failed"
]);

function numeric(value) {
  return Number.isFinite(Number(value)) ? Math.max(0, Number(value)) : 0;
}

export function evaluateTaskResult({
  task,
  agentRun = {},
  independentCheck = {},
  protectedIntegrity = {},
  sourceChanged = false,
  elapsedMs = 0,
  peakControllerRssBytes = null,
  modelMemory = null
}) {
  const attempts = Array.isArray(agentRun.attempts) ? agentRun.attempts : [];
  const changedFiles = Array.isArray(agentRun.changedFiles) ? agentRun.changedFiles : [];
  const changedPaths = changedFiles.map(item => item?.path).filter(value => typeof value === "string");
  const patchValid = attempts.some(item => VALID_PATCH_ATTEMPT_STATES.has(item?.status));
  const agentVerificationPassed = agentRun.verification?.passed === true;
  const independentTestPassed = independentCheck.ok === true && independentCheck.output?.passed === true;
  const protectedFilesIntact = Object.values(protectedIntegrity).every(value => value === true);
  const onlyExpectedFilesChanged = changedPaths.length > 0
    && changedPaths.every(relative => relative === task.sourceFile);
  const taskCompleted = patchValid && agentVerificationPassed && independentTestPassed
    && sourceChanged && protectedFilesIntact && onlyExpectedFilesChanged;

  const usage = agentRun.modelUsage && typeof agentRun.modelUsage === "object" ? agentRun.modelUsage : null;
  const promptTokens = usage
    ? numeric(usage.promptTokens)
    : attempts.reduce((sum, item) => sum + numeric(item?.promptTokens), 0);
  const outputTokens = usage
    ? numeric(usage.outputTokens)
    : attempts.reduce((sum, item) => sum + numeric(item?.outputTokens), 0);
  const modelMemoryBefore = modelMemory?.before || null;
  const modelMemoryAfter = modelMemory?.after || null;
  const modelVramValues = [modelMemoryBefore?.vramBytes, modelMemoryAfter?.vramBytes]
    .filter(value => Number.isFinite(value));
  const failureReasons = [];
  if (!patchValid) failureReasons.push("No schema-valid patch attempt was recorded.");
  if (!agentVerificationPassed) failureReasons.push("The agent's own verification did not pass.");
  if (!independentTestPassed) failureReasons.push("The independent post-run test command did not pass.");
  if (!sourceChanged) failureReasons.push("The expected target source file was not changed.");
  if (!protectedFilesIntact) failureReasons.push("A protected benchmark fixture file was modified or removed.");
  if (!onlyExpectedFilesChanged) failureReasons.push("The agent changed an unexpected file or produced no changed-file record.");

  return {
    patchValid,
    agentVerificationPassed,
    independentTestPassed,
    sourceChanged,
    protectedFilesIntact,
    onlyExpectedFilesChanged,
    taskCompleted,
    elapsedMs: numeric(elapsedMs),
    attempts: attempts.length,
    repairAttempts: Math.max(0, attempts.length - 1),
    promptTokens,
    outputTokens,
    totalTokens: promptTokens + outputTokens,
    generationCalls: usage ? numeric(usage.generationCalls) : null,
    promptTokenCountsExact: usage ? usage.promptTokenCountsExact === true : false,
    tokenUsageSource: usage ? "agent-model-request-telemetry" : "patch-attempt-fallback",
    peakControllerRssBytes: Number.isFinite(Number(peakControllerRssBytes))
      ? Number(peakControllerRssBytes) : null,
    modelVramBytesBefore: modelVramValues.length ? modelMemoryBefore?.vramBytes ?? null : null,
    modelVramBytesAfter: modelVramValues.length ? modelMemoryAfter?.vramBytes ?? null : null,
    modelMemorySource: modelMemoryAfter?.available ? modelMemoryAfter.source : null,
    modelMemoryScope: modelMemoryAfter?.scope || modelMemoryBefore?.scope || "not exposed by the current provider API",
    changedPaths,
    failureReasons
  };
}

function average(values) {
  const usable = values.filter(value => Number.isFinite(value));
  return usable.length ? usable.reduce((sum, value) => sum + value, 0) / usable.length : null;
}

function median(values) {
  const usable = values.filter(value => Number.isFinite(value)).sort((a, b) => a - b);
  if (!usable.length) return null;
  const middle = Math.floor(usable.length / 2);
  return usable.length % 2 ? usable[middle] : (usable[middle - 1] + usable[middle]) / 2;
}

export function summarizeBenchmark(results = []) {
  const tasks = Array.isArray(results) ? results : [];
  const completed = tasks.filter(item => item.metrics?.taskCompleted === true).length;
  const patchValid = tasks.filter(item => item.metrics?.patchValid === true).length;
  const independentTestsPassed = tasks.filter(item => item.metrics?.independentTestPassed === true).length;
  const latencies = tasks.map(item => item.metrics?.elapsedMs).filter(Number.isFinite);
  const tokens = tasks.map(item => item.metrics?.totalTokens).filter(Number.isFinite);
  const byModel = new Map();
  for (const item of tasks) {
    const key = [item.model?.provider || "unknown", item.model?.name || "unknown"].join(" / ");
    if (!byModel.has(key)) byModel.set(key, []);
    byModel.get(key).push(item);
  }
  return {
    taskCount: tasks.length,
    completed,
    completionRate: tasks.length ? completed / tasks.length : 0,
    patchValid,
    patchValidityRate: tasks.length ? patchValid / tasks.length : 0,
    independentTestsPassed,
    independentTestPassRate: tasks.length ? independentTestsPassed / tasks.length : 0,
    averageLatencyMs: average(latencies),
    medianLatencyMs: median(latencies),
    totalTokens: tokens.reduce((sum, value) => sum + value, 0),
    averageTokens: average(tokens),
    byModel: [...byModel.entries()].map(([model, items]) => ({
      model,
      taskCount: items.length,
      completed: items.filter(item => item.metrics?.taskCompleted === true).length,
      completionRate: items.length
        ? items.filter(item => item.metrics?.taskCompleted === true).length / items.length : 0,
      totalTokens: items.reduce((sum, item) => sum + numeric(item.metrics?.totalTokens), 0),
      averageLatencyMs: average(items.map(item => item.metrics?.elapsedMs))
    }))
  };
}
