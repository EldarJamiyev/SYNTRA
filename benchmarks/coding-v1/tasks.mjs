export const CODING_BENCHMARK_VERSION = "coding-v1";
export const CODING_BENCHMARK_SCHEMA_VERSION = 1;

export const CODING_BENCHMARK_TASKS = [
  {
    "id": "slugify-normalization",
    "category": "string-processing",
    "title": "Normalize human-readable text into a URL slug",
    "sourceFile": "src/slugify.js",
    "protectedFiles": [
      "package.json",
      "tests/slugify.test.js"
    ],
    "prompt": "Implement the slugify(value) function in src/slugify.js. It must return an ASCII lowercase URL slug: normalize Unicode accents with NFKD and remove combining marks, convert every run of non-ASCII-alphanumeric characters into one hyphen, and trim leading/trailing hyphens. Treat null and undefined as an empty string. Preserve the exported function name. Do not change tests or package.json.",
    "files": {
      "package.json": "{\n  \"name\": \"syntra-coding-benchmark-fixture\",\n  \"private\": true,\n  \"type\": \"module\",\n  \"scripts\": {\n    \"test\": \"node --test\"\n  }\n}\n",
      "src/slugify.js": "export function slugify(value) {\n  return String(value ?? \"\").toLowerCase();\n}\n",
      "tests/slugify.test.js": "import assert from \"node:assert/strict\";\nimport test from \"node:test\";\nimport { slugify } from \"../src/slugify.js\";\n\ntest(\"lowercases and separates words\", () => {\n  assert.equal(slugify(\"Hello World\"), \"hello-world\");\n});\n\ntest(\"normalizes accents and repeated punctuation\", () => {\n  assert.equal(slugify(\"  Crème & Brûlée!!!  \"), \"creme-brulee\");\n});\n\ntest(\"collapses separators and trims hyphens\", () => {\n  assert.equal(slugify(\"--API___Gateway / v2--\"), \"api-gateway-v2\");\n});\n\ntest(\"handles nullish and non-string values\", () => {\n  assert.equal(slugify(null), \"\");\n  assert.equal(slugify(undefined), \"\");\n  assert.equal(slugify(42), \"42\");\n});\n"
    }
  },
  {
    "id": "duration-parser",
    "category": "input-validation",
    "title": "Parse bounded human-readable durations",
    "sourceFile": "src/parse-duration.js",
    "protectedFiles": [
      "package.json",
      "tests/parse-duration.test.js"
    ],
    "prompt": "Implement parseDuration(value) in src/parse-duration.js. Accept a string containing a non-negative decimal number and one unit: ms, s, m, or h (case-insensitive), with optional surrounding whitespace. Return the exact duration in integer milliseconds; reject fractional millisecond results, empty strings, negative values, missing units, unknown units, non-finite values, and durations greater than 30 days by returning null. Do not throw for invalid input. Preserve the export name and do not change tests or package.json.",
    "files": {
      "package.json": "{\n  \"name\": \"syntra-coding-benchmark-fixture\",\n  \"private\": true,\n  \"type\": \"module\",\n  \"scripts\": {\n    \"test\": \"node --test\"\n  }\n}\n",
      "src/parse-duration.js": "export function parseDuration(value) {\n  return Number(value);\n}\n",
      "tests/parse-duration.test.js": "import assert from \"node:assert/strict\";\nimport test from \"node:test\";\nimport { parseDuration } from \"../src/parse-duration.js\";\n\ntest(\"parses supported units and decimals\", () => {\n  assert.equal(parseDuration(\"250ms\"), 250);\n  assert.equal(parseDuration(\"2s\"), 2000);\n  assert.equal(parseDuration(\"1.5m\"), 90000);\n  assert.equal(parseDuration(\"1h\"), 3600000);\n  assert.equal(parseDuration(\" 2 S \"), 2000);\n});\n\ntest(\"rejects malformed and negative values\", () => {\n  for (const value of [\"\", \"10\", \"-1s\", \"1d\", \"1.2.3s\", \"NaNs\", null, 200]) {\n    assert.equal(parseDuration(value), null, String(value));\n  }\n});\n\ntest(\"rejects fractional milliseconds and durations above 30 days\", () => {\n  assert.equal(parseDuration(\"0.1ms\"), null);\n  assert.equal(parseDuration(\"720h\"), 2592000000);\n  assert.equal(parseDuration(\"720h1ms\"), null);\n});\n"
    }
  },
  {
    "id": "case-insensitive-header-merge",
    "category": "data-structures",
    "title": "Merge HTTP headers without case-sensitive duplicates",
    "sourceFile": "src/merge-headers.js",
    "protectedFiles": [
      "package.json",
      "tests/merge-headers.test.js"
    ],
    "prompt": "Implement mergeHeaders(...sources) in src/merge-headers.js. Accept any number of plain object sources; ignore null, arrays, and non-plain objects. Normalize header names by trimming and lowercasing them. Ignore empty names and null/undefined values. Convert accepted values to strings and trim their outside whitespace. Merge case-insensitively so a later source overrides an earlier value even when the header casing differs. Return a plain object. Do not mutate the source objects or change tests/package.json.",
    "files": {
      "package.json": "{\n  \"name\": \"syntra-coding-benchmark-fixture\",\n  \"private\": true,\n  \"type\": \"module\",\n  \"scripts\": {\n    \"test\": \"node --test\"\n  }\n}\n",
      "src/merge-headers.js": "export function mergeHeaders(...sources) {\n  return Object.assign({}, ...sources);\n}\n",
      "tests/merge-headers.test.js": "import assert from \"node:assert/strict\";\nimport test from \"node:test\";\nimport { mergeHeaders } from \"../src/merge-headers.js\";\n\ntest(\"merges header names without case-sensitive duplicates\", () => {\n  assert.deepEqual(\n    mergeHeaders(\n      { Accept: \"application/json\", \"X-Trace-Id\": \"old\" },\n      { \"x-trace-id\": \"new\", Authorization: \"Bearer local\" }\n    ),\n    {\n      accept: \"application/json\",\n      \"x-trace-id\": \"new\",\n      authorization: \"Bearer local\"\n    }\n  );\n});\n\ntest(\"trims names and values and ignores invalid entries\", () => {\n  assert.deepEqual(\n    mergeHeaders(null, [\"bad\"], { \" X-Mode \": \" fast \", \"\": \"skip\", Missing: null, \"X-Count\": 3 }),\n    { \"x-mode\": \"fast\", \"x-count\": \"3\" }\n  );\n});\n\ntest(\"does not mutate source objects\", () => {\n  const source = { \"X-Name\": \"value\" };\n  mergeHeaders(source, { \"x-name\": \"next\" });\n  assert.deepEqual(source, { \"X-Name\": \"value\" });\n});\n\ntest(\"ignores non-plain object sources\", () => {\n  assert.deepEqual(mergeHeaders(new Map([[\"X-Map\", \"ignored\"]]), { \"X-Ok\": \"yes\" }), { \"x-ok\": \"yes\" });\n});\n"
    }
  }
];

export function validateBenchmarkManifest(input = CODING_BENCHMARK_TASKS) {
  if (!Array.isArray(input) || input.length < 1 || input.length > 20) {
    throw new Error("The coding benchmark must contain 1-20 tasks.");
  }
  const seen = new Set();
  for (const task of input) {
    if (!task || typeof task !== "object" || !/^[a-z0-9][a-z0-9-]{1,63}$/.test(task.id || "")
      || seen.has(task.id) || typeof task.prompt !== "string" || task.prompt.length < 20
      || typeof task.sourceFile !== "string" || !task.sourceFile.startsWith("src/")
      || !task.files || typeof task.files !== "object" || Array.isArray(task.files)) {
      throw new Error("A coding benchmark task has an invalid identifier, prompt, source path, or fixture map.");
    }
    seen.add(task.id);
    for (const [relative, content] of Object.entries(task.files)) {
      if (relative.startsWith("/") || relative.includes("\\\\") || relative.split("/").some(part => !part || part === "." || part === "..")
        || typeof content !== "string" || Buffer.byteLength(content, "utf8") > 128 * 1024) {
        throw new Error("Benchmark fixture files must be bounded UTF-8 files beneath their temporary workspace.");
      }
    }
    if (!Object.hasOwn(task.files, task.sourceFile)) {
      throw new Error("Benchmark source file is not present in the fixture.");
    }
    for (const protectedPath of task.protectedFiles || []) {
      if (!Object.hasOwn(task.files, protectedPath)) throw new Error("A protected benchmark fixture file is missing.");
    }
  }
  return true;
}

validateBenchmarkManifest();
