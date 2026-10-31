import assert from "node:assert/strict";
import test from "node:test";
import { slugify } from "../src/strings.mjs";
test("slugify removes punctuation and joins words", () => assert.equal(slugify("Hello, World!"), "hello-world"));
test("slugify collapses separators and trims the result", () => assert.equal(slugify("  --One   Two---  "), "one-two"));
test("slugify preserves empty input behavior", () => {
  assert.equal(slugify(null), "");
  assert.equal(slugify(undefined), "");
  assert.equal(slugify("!!!"), "");
});
