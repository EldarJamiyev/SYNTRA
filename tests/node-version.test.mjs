import assert from "node:assert/strict";
import test from "node:test";

import {
  getNodeVersionError,
  isSupportedNodeVersion,
  parseNodeVersion,
  SUPPORTED_NODE_RANGE
} from "../scripts/node-version.mjs";

test("SYNTRA accepts the minimum supported Node 22 runtime and newer major versions", () => {
  assert.equal(SUPPORTED_NODE_RANGE, ">=22.12.0");
  for (const version of [
    "22.12.0",
    "v22.12.0",
    "22.12.1",
    "22.13.0",
    "22.20.0",
    "24.0.0",
    "24.11.1",
    "26.0.0",
    "27.0.0"
  ]) {
    assert.equal(isSupportedNodeVersion(version), true, version);
  }
});

test("SYNTRA rejects versions below the Vite 8 runtime floor and prereleases", () => {
  for (const version of [
    "18.20.8",
    "20.19.0",
    "21.7.3",
    "22.0.0",
    "22.11.9",
    "22.12.0-rc.1",
    "24.0.0-alpha.1",
    "27.0.0-alpha.1",
    "v22",
    "22.12",
    "",
    null,
    undefined
  ]) {
    assert.equal(isSupportedNodeVersion(version), false, String(version));
  }
});

test("Node version parsing preserves numeric components and prerelease identity", () => {
  assert.deepEqual(parseNodeVersion("v22.12.3"), {
    major: 22,
    minor: 12,
    patch: 3,
    prerelease: null
  });
  assert.deepEqual(parseNodeVersion("26.0.0-rc.2"), {
    major: 26,
    minor: 0,
    patch: 0,
    prerelease: "rc.2"
  });
  assert.equal(parseNodeVersion("22.12"), null);
  assert.equal(parseNodeVersion("22.12.0.1"), null);
});

test("unsupported-version diagnostic states the accepted engine range and detected version", () => {
  assert.equal(
    getNodeVersionError("v20.19.0"),
    "SYNTRA requires Node.js >=22.12.0 because its Vite 8 frontend does not support older runtimes. Detected: v20.19.0."
  );
  assert.match(getNodeVersionError(""), /Detected: unknown/u);
});
