import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

function diagnosticText(ts, diagnostics) {
  return ts.formatDiagnosticsWithColorAndContext(diagnostics, {
    getCurrentDirectory: () => process.cwd(),
    getCanonicalFileName: value => value,
    getNewLine: () => "\n"
  });
}

async function main() {
  const [inputFile, outputDirectory] = process.argv.slice(2);
  if (!inputFile || !outputDirectory) {
    throw new Error("Usage: typescript_test_runner.mjs <test.ts> <output-directory>");
  }

  const require = createRequire(import.meta.url);
  const typescriptRoot = process.env.SYNTRA_TYPESCRIPT_ROOT || "/opt/typescript";
  const ts = require(typescriptRoot);
  const testFile = path.resolve(inputFile);
  const outDir = path.resolve(outputDirectory);
  const filesystemRoot = path.parse(testFile).root;

  const options = {
    strict: true,
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    skipLibCheck: true,
    rootDir: filesystemRoot,
    outDir,
    noEmitOnError: true,
    forceConsistentCasingInFileNames: true
  };

  const program = ts.createProgram([testFile], options);
  const diagnostics = ts.getPreEmitDiagnostics(program);
  if (diagnostics.length) {
    process.stderr.write("SYNTRA_BENCHMARK_ASSERTION_FAILED: TypeScript compile-time contract failed.\n");
    process.stderr.write(diagnosticText(ts, diagnostics));
    process.exitCode = 1;
    return;
  }

  const emitted = program.emit();
  if (emitted.emitSkipped) {
    throw new Error("TypeScript refused to emit the validated hidden test.");
  }

  await fs.mkdir(outDir, { recursive: true });
  await fs.writeFile(path.join(outDir, "package.json"), JSON.stringify({ type: "module" }), "utf8");
  const relativeTest = path.relative(filesystemRoot, testFile).replace(/\.tsx?$/i, ".js");
  const emittedTest = path.join(outDir, relativeTest);
  try {
    await import(pathToFileURL(emittedTest).href);
  } catch (error) {
    process.stderr.write("SYNTRA_BENCHMARK_ASSERTION_FAILED: TypeScript runtime assertion failed.\n");
    process.stderr.write((error instanceof Error ? error.stack : String(error)) + "\n");
    process.exitCode = 1;
  }
}

main().catch(error => {
  process.stderr.write((error instanceof Error ? error.stack : String(error)) + "\n");
  process.exitCode = 1;
});
