import assert from "node:assert/strict";
import test from "node:test";

import { countByteBpe } from "../local/brain-runtime.mjs";

test("the workspace API BPE counter follows the checkpoint merge ranks exactly", () => {
  const tokenizer = {
    format: "syntra-local-byte-bpe-v1",
    vocab_size: 262,
    base_vocab_size: 258,
    merges: [
      [104, 101],
      [108, 108],
      [258, 259],
      [260, 111]
    ]
  };
  assert.equal(countByteBpe("hello", tokenizer), 1);
  assert.equal(countByteBpe("hello!", tokenizer), 2);
  assert.equal(countByteBpe("hello hello", tokenizer), 3);
  assert.equal(countByteBpe("", tokenizer), 0);
});
