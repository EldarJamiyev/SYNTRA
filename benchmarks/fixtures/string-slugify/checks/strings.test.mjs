import assert from "node:assert/strict";
import { slugify } from "../src/strings.mjs";
assert.equal(slugify("Hello, World!"), "hello-world");
assert.equal(slugify("  --One   Two---  "), "one-two");
assert.equal(slugify(null), "");
assert.equal(slugify(undefined), "");
assert.equal(slugify("!!!"), "");
