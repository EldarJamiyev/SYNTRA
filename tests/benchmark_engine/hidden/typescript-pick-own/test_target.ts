import { pickOwn } from "../workspace/src/pick-own.js";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

type RecordShape = { own: string; inherited: string; optional?: string };
const prototype = { inherited: "prototype-value" };
const source = Object.assign(Object.create(prototype), {
  own: "own-value",
  optional: undefined,
}) as RecordShape;

const picked = pickOwn(source, ["own", "inherited", "optional"] as const);
assert(picked.own === "own-value", "Own string properties must be copied.");
assert(!Object.hasOwn(picked, "inherited"), "Inherited properties must not be materialized as own output properties.");
assert(Object.hasOwn(picked, "optional"), "An own property with undefined must be preserved.");
assert(picked.optional === undefined, "The undefined own value must remain unchanged.");
assert(Object.keys(picked).sort().join(",") === "optional,own", "Only selected own properties may be emitted.");
assert(source.inherited === "prototype-value", "The source prototype must remain unchanged.");

// @ts-expect-error Keys absent from the source type must be rejected.
pickOwn(source, ["missing"] as const);
