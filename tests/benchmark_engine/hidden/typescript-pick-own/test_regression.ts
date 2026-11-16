import { pickOwn } from "../workspace/src/pick-own.js";

const source = { name: "Ada", count: 3, enabled: true };
const result = pickOwn(source, ["name", "count"] as const);
if (result.name !== "Ada" || result.count !== 3 || Object.keys(result).length !== 2) {
  throw new Error("Own-property selection regression.");
}
