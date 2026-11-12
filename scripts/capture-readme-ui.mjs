#!/usr/bin/env node
import fs from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "artifacts", "readme", "syntra-chat-ui.png");
const children = [];
let browser;

function start(command, args, env = {}) {
  const child = spawn(command, args, {
    cwd: root,
    env: { ...process.env, ...env },
    shell: false,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"]
  });
  let recentOutput = "";
  child.stdout.on("data", chunk => { recentOutput = (recentOutput + chunk.toString("utf8")).slice(-4000); });
  child.stderr.on("data", chunk => { recentOutput = (recentOutput + chunk.toString("utf8")).slice(-4000); });
  child.syntraOutput = () => recentOutput;
  children.push(child);
  return child;
}

async function waitForService(child, url, label, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = "not checked";
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(label + " exited before becoming ready. " + child.syntraOutput());
    }
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(1500) });
      if (response.ok) return;
      lastError = "HTTP " + response.status;
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    }
    await new Promise(resolve => setTimeout(resolve, 350));
  }
  throw new Error(label + " did not become ready: " + lastError + ". " + child.syntraOutput());
}

async function stopChildren() {
  for (const child of [...children].reverse()) {
    if (child.exitCode === null && !child.killed) child.kill("SIGTERM");
  }
  await Promise.all(children.map(child => new Promise(resolve => {
    if (child.exitCode !== null) return resolve();
    const timer = setTimeout(() => {
      if (child.exitCode === null) child.kill("SIGKILL");
      resolve();
    }, 3000);
    child.once("exit", () => { clearTimeout(timer); resolve(); });
  })));
}

try {
  const localApi = start(process.execPath, ["local/server.mjs"], {
    SYNTRA_LOCAL_PORT: "3001",
    SYNTRA_MODEL_PROVIDER: "native"
  });
  const frontend = start(process.execPath, [
    "node_modules/vite/bin/vite.js", "--host", "127.0.0.1", "--port", "3000", "--strictPort"
  ]);
  await waitForService(localApi, "http://127.0.0.1:3001/local-api/health", "SYNTRA workspace API");
  await waitForService(frontend, "http://127.0.0.1:3000/", "SYNTRA frontend");

  await fs.mkdir(path.dirname(output), { recursive: true });
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1
  });
  const pageErrors = [];
  page.on("pageerror", error => pageErrors.push(error.message));
  const response = await page.goto("http://127.0.0.1:3000/", { waitUntil: "domcontentloaded", timeout: 20000 });
  if (!response?.ok()) throw new Error("SYNTRA UI returned HTTP " + (response?.status() ?? "no response"));
  await page.waitForSelector("main.chat-app", { timeout: 15000 });
  await page.getByRole("textbox").first().waitFor({ state: "visible", timeout: 10000 });
  await page.screenshot({ path: output, fullPage: true, animations: "disabled" });
  if (pageErrors.length) throw new Error("Browser console page errors: " + pageErrors.join(" | "));
  const stat = await fs.stat(output);
  if (stat.size < 10000) throw new Error("The captured UI screenshot is unexpectedly small; refusing to publish it.");
  process.stdout.write("Captured actual SYNTRA React UI screenshot: " + path.relative(root, output) + " (" + stat.size + " bytes)\n");
} catch (error) {
  process.stderr.write("SYNTRA UI capture failed: " + (error instanceof Error ? error.message : String(error)) + "\n");
  process.exitCode = 1;
} finally {
  if (browser) await browser.close().catch(() => {});
  await stopChildren();
}
