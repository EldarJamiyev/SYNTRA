#!/usr/bin/env node

import fs from "node:fs/promises";
import {
  applyFileProposal,
  createProjectPlan,
  inspectWorkspace,
  listProposals,
  proposeFileChange,
  readWorkspaceFile,
  runProjectCheck,
  searchWorkspace
} from "../local/workspace.mjs";

function parseArgs(argv) {
  const result = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const item = argv[i];
    if (item.startsWith("--")) {
      const name = item.slice(2);
      const next = argv[i + 1];
      if (next && !next.startsWith("--")) {
        result[name] = next;
        i++;
      } else {
        result[name] = true;
      }
    } else {
      result._.push(item);
    }
  }
  return result;
}

function help() {
  console.log([
    "SYNTRA local project tools",
    "",
    "Commands:",
    "  node bin/syntra.mjs inspect --root <absolute-path>",
    "  node bin/syntra.mjs search --root <absolute-path> --query <terms>",
    "  node bin/syntra.mjs read --root <absolute-path> --file <relative-path>",
    "  node bin/syntra.mjs plan --root <absolute-path> --goal <description>",
    "  node bin/syntra.mjs proposals --root <absolute-path>",
    "  node bin/syntra.mjs propose --root <absolute-path> --file <relative-path> --content-file <path>",
    "  node bin/syntra.mjs apply --root <absolute-path> --proposal <uuid> --confirm",
    "  node bin/syntra.mjs check --root <absolute-path> --task build|test|lint|typecheck",
    "",
    "Checks use project-defined scripts and execute code in the selected project.",
    "No external AI service or model API is used by these workspace tools."
  ].join("\n"));
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const command = args._[0];
  if (!command || command === "help" || command === "--help") {
    help();
    return;
  }
  let result;
  if (command === "inspect") result = await inspectWorkspace(args.root);
  else if (command === "search") result = await searchWorkspace(args.root, args.query, Number(args.limit || 20));
  else if (command === "read") result = await readWorkspaceFile(args.root, args.file);
  else if (command === "plan") result = await createProjectPlan(args.root, args.goal);
  else if (command === "proposals") result = await listProposals(args.root);
  else if (command === "propose") {
    if (!args["content-file"]) throw new Error("--content-file is required.");
    const content = await fs.readFile(args["content-file"], "utf8");
    result = await proposeFileChange(args.root, args.file, content);
  } else if (command === "apply") {
    result = await applyFileProposal(args.root, args.proposal, args.confirm === true);
  } else if (command === "check") {
    result = await runProjectCheck(args.root, args.task, { timeoutMs: Number(args.timeout || 120000) });
  } else {
    throw new Error("Unknown command: " + command + ". Run help for supported commands.");
  }
  console.log(JSON.stringify(result, null, 2));
  if (result && result.passed === false) process.exitCode = 1;
}

main().catch(error => {
  console.error("SYNTRA: " + (error instanceof Error ? error.message : String(error)));
  process.exitCode = 1;
});
