const SEMVER = /^\d+\.\d+\.\d+$/;

function compareSemver(left, right) {
  const a = left.split(".").map(Number);
  const b = right.split(".").map(Number);
  for (let index = 0; index < 3; index += 1) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return 0;
}

function finiteNonNegative(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

export function validateBenchmarkReport(report) {
  if (!report || typeof report !== "object" || Array.isArray(report)) {
    throw new Error("Benchmark report must be a JSON object.");
  }
  if (report.schemaVersion !== 1) throw new Error("Unsupported benchmark report schema version.");
  if (typeof report.suiteVersion !== "string" || !SEMVER.test(report.suiteVersion)) {
    throw new Error("Benchmark report has an invalid suite version.");
  }
  if (typeof report.runId !== "string" || !report.runId.trim()) {
    throw new Error("Benchmark report is missing its run ID.");
  }
  if (!report.model || typeof report.model.name !== "string" || !report.model.name.trim()) {
    throw new Error("Benchmark report is missing its model identity.");
  }
  if (!Array.isArray(report.tasks) || report.tasks.length < 1 || report.tasks.length > 100) {
    throw new Error("Benchmark report must contain between 1 and 100 task results.");
  }
  const ids = new Set();
  for (const task of report.tasks) {
    if (!task || typeof task.id !== "string" || !/^[a-z0-9][a-z0-9-]{1,63}$/.test(task.id) || ids.has(task.id)) {
      throw new Error("Benchmark report contains an invalid or duplicate task ID.");
    }
    ids.add(task.id);
    for (const flag of ["patchValid", "verificationPassed", "taskComplete"]) {
      if (typeof task[flag] !== "boolean") throw new Error("Benchmark task " + task.id + " has an invalid " + flag + " value.");
    }
    if (!finiteNonNegative(task.latencyMs)) throw new Error("Benchmark task " + task.id + " has an invalid latency value.");
    if (!task.tokens || !finiteNonNegative(task.tokens.prompt) || !finiteNonNegative(task.tokens.output)
      || !finiteNonNegative(task.tokens.total)) {
      throw new Error("Benchmark task " + task.id + " has invalid token measurements.");
    }
    const baselineValidationRequired = compareSemver(report.suiteVersion, "1.4.0") >= 0;
    if (baselineValidationRequired && typeof task.baselineValidated !== "boolean") {
      throw new Error("Benchmark task " + task.id + " is missing required baseline validation evidence.");
    }
    if (task.baselineValidated !== undefined && typeof task.baselineValidated !== "boolean") {
      throw new Error("Benchmark task " + task.id + " has an invalid baselineValidated value.");
    }
    const expectedComplete = task.patchValid && task.verificationPassed
      && task.baselineValidated !== false;
    if (task.taskComplete !== expectedComplete) {
      throw new Error("Benchmark task " + task.id + " has an inconsistent completion status.");
    }
    if (task.tokens.total !== task.tokens.prompt + task.tokens.output) {
      throw new Error("Benchmark task " + task.id + " has inconsistent token totals.");
    }
    if (!finiteNonNegative(task.peakProcessRssBytes)) {
      throw new Error("Benchmark task " + task.id + " has an invalid RSS measurement.");
    }
  }
  const summary = report.summary;
  if (!summary || summary.taskCount !== report.tasks.length
    || !finiteNonNegative(summary.taskCompletionRate) || summary.taskCompletionRate > 1
    || !finiteNonNegative(summary.patchValidityRate) || summary.patchValidityRate > 1
    || !finiteNonNegative(summary.verificationPassRate) || summary.verificationPassRate > 1
    || !summary.latencyMs || !finiteNonNegative(summary.latencyMs.mean)
    || !summary.tokens || !finiteNonNegative(summary.tokens.total)
    || !finiteNonNegative(summary.peakProcessRssBytes)) {
    throw new Error("Benchmark report summary is incomplete or invalid.");
  }
  return report;
}

function metricDelta(key, baseline, candidate, higherIsBetter) {
  const delta = candidate - baseline;
  return {
    key, baseline, candidate, delta,
    relativeChangePct: baseline === 0 ? null : Number(((delta / baseline) * 100).toFixed(2)),
    direction: delta === 0 ? "unchanged" : (delta > 0) === higherIsBetter ? "improved" : "regressed"
  };
}

function taskMap(report) {
  return new Map(report.tasks.map(task => [task.id, task]));
}

export function compareBenchmarkReports(baselineInput, candidateInput) {
  const baseline = validateBenchmarkReport(baselineInput);
  const candidate = validateBenchmarkReport(candidateInput);
  if (baseline.suiteVersion !== candidate.suiteVersion) {
    throw new Error("Cannot compare suite versions " + baseline.suiteVersion + " and " + candidate.suiteVersion + ". Run the same benchmark suite version.");
  }
  const baselineTasks = taskMap(baseline);
  const candidateTasks = taskMap(candidate);
  const baselineIds = [...baselineTasks.keys()].sort();
  const candidateIds = [...candidateTasks.keys()].sort();
  if (baselineIds.length !== candidateIds.length || baselineIds.some((id, index) => id !== candidateIds[index])) {
    throw new Error("Cannot compare different task sets. Both reports must contain exactly the same benchmark task IDs.");
  }

  const metrics = [
    metricDelta("taskCompletionRate", baseline.summary.taskCompletionRate, candidate.summary.taskCompletionRate, true),
    metricDelta("patchValidityRate", baseline.summary.patchValidityRate, candidate.summary.patchValidityRate, true),
    metricDelta("verificationPassRate", baseline.summary.verificationPassRate, candidate.summary.verificationPassRate, true),
    metricDelta("meanLatencyMs", baseline.summary.latencyMs.mean, candidate.summary.latencyMs.mean, false),
    metricDelta("totalTokens", baseline.summary.tokens.total, candidate.summary.tokens.total, false),
    metricDelta("peakProcessRssBytes", baseline.summary.peakProcessRssBytes, candidate.summary.peakProcessRssBytes, false)
  ];
  const tasks = baselineIds.map(id => {
    const before = baselineTasks.get(id);
    const after = candidateTasks.get(id);
    const changes = [];
    for (const field of ["patchValid", "verificationPassed", "taskComplete"]) {
      if (before[field] !== after[field]) {
        changes.push({ field, baseline: before[field], candidate: after[field], direction: after[field] ? "improved" : "regressed" });
      }
    }
    const latency = metricDelta("latencyMs", before.latencyMs, after.latencyMs, false);
    return {
      id, title: after.title || before.title || id,
      baselineComplete: before.taskComplete,
      candidateComplete: after.taskComplete,
      status: changes.some(change => change.direction === "regressed") ? "regressed"
        : changes.some(change => change.direction === "improved") ? "improved" : "unchanged",
      changes, latency
    };
  });

  const baselineRuntime = baseline.runtime || {};
  const candidateRuntime = candidate.runtime || {};
  const baselineHardware = baseline.hardware || {};
  const candidateHardware = candidate.hardware || {};
  const warnings = [];
  const runtimeComparable = baselineRuntime.platform === candidateRuntime.platform
    && baselineRuntime.architecture === candidateRuntime.architecture
    && String(baselineRuntime.node || "").split(".")[0] === String(candidateRuntime.node || "").split(".")[0];
  const hardwareMetadataPresent = typeof baselineHardware.cpuModel === "string"
    && baselineHardware.cpuModel.length > 0
    && typeof candidateHardware.cpuModel === "string" && candidateHardware.cpuModel.length > 0
    && finiteNonNegative(baselineHardware.totalMemoryBytes) && baselineHardware.totalMemoryBytes > 0
    && finiteNonNegative(candidateHardware.totalMemoryBytes) && candidateHardware.totalMemoryBytes > 0
    && finiteNonNegative(baselineHardware.logicalCores) && baselineHardware.logicalCores > 0
    && finiteNonNegative(candidateHardware.logicalCores) && candidateHardware.logicalCores > 0;
  const hardwareComparable = hardwareMetadataPresent
    && baselineHardware.cpuModel === candidateHardware.cpuModel
    && baselineHardware.totalMemoryBytes === candidateHardware.totalMemoryBytes
    && baselineHardware.logicalCores === candidateHardware.logicalCores;
  if (!runtimeComparable) warnings.push("Runtime platforms, architectures, or Node.js major versions differ. Interpret latency and process-memory deltas cautiously.");
  if (!hardwareComparable) warnings.push("CPU model, logical-core count, or installed-memory metadata is missing or differs. Latency and process-memory deltas are not a controlled hardware comparison.");
  const mixedEnvironment = !runtimeComparable || !hardwareComparable;
  if (mixedEnvironment) {
    for (const metric of metrics) {
      if (metric.key === "meanLatencyMs" || metric.key === "peakProcessRssBytes") metric.direction = "incomparable";
    }
    for (const task of tasks) task.latency.direction = "incomparable";
  }

  const qualityDirections = metrics.slice(0, 3).map(item => item.direction);
  const performanceDirections = metrics.slice(3).map(item => item.direction);
  const allDirections = [...qualityDirections, ...performanceDirections].filter(value => value !== "incomparable");
  const improvedCount = allDirections.filter(value => value === "improved").length;
  const regressedCount = allDirections.filter(value => value === "regressed").length;
  const status = regressedCount && improvedCount ? "mixed"
    : regressedCount ? "regressed"
      : improvedCount ? "improved" : "unchanged";

  return {
    schemaVersion: 1,
    suiteVersion: baseline.suiteVersion,
    comparedAt: new Date().toISOString(),
    baseline: {
      runId: baseline.runId, model: baseline.model.name, startedAt: baseline.startedAt || null,
      completedTaskCount: baseline.summary.completedTaskCount,
      taskCount: baseline.summary.taskCount
    },
    candidate: {
      runId: candidate.runId, model: candidate.model.name, startedAt: candidate.startedAt || null,
      completedTaskCount: candidate.summary.completedTaskCount,
      taskCount: candidate.summary.taskCount
    },
    status, metrics, tasks, warnings,
    environment: { runtimeComparable, hardwareComparable }
  };
}
