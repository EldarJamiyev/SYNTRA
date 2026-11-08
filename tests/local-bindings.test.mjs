import assert from "node:assert/strict";
import fs from "node:fs/promises";
import test from "node:test";

const packagePath = new URL("../package.json", import.meta.url);
const viteConfigPath = new URL("../vite.config.ts", import.meta.url);

test("Vite development and preview servers are restricted to loopback", async () => {
  const [packageText, viteConfig] = await Promise.all([
    fs.readFile(packagePath, "utf8"),
    fs.readFile(viteConfigPath, "utf8"),
  ]);
  const scripts = JSON.parse(packageText).scripts;

  assert.match(scripts.dev, /vite --host 127\.0\.0\.1 --port 3000/);
  assert.match(scripts.preview, /vite preview --host 127\.0\.0\.1 --port 3000/);
  assert.doesNotMatch(scripts.dev + scripts.preview, /0\.0\.0\.0|--host\s+\*/);

  assert.equal((viteConfig.match(/host: '127\.0\.0\.1'/g) || []).length, 2);
  assert.equal(
    (viteConfig.match(/allowedHosts: loopbackHosts/g) || []).length,
    2,
    "both development and preview servers must enforce the explicit loopback host allowlist",
  );
  assert.match(viteConfig, /const loopbackHosts = \['localhost', '127\.0\.0\.1'\]/);
  assert.doesNotMatch(viteConfig, /0\.0\.0\.0|allowedHosts:\s*true/);
});
