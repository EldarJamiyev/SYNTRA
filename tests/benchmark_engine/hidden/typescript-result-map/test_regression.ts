import { mapResult } from "../workspace/src/map-result.js";

const result = mapResult({ ok: true as const, value: 5 }, value => value * 2);
if (!result.ok || result.value !== 10) {
  throw new Error("Successful result mapping regression.");
}
