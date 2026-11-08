import { mapObject } from "../workspace/src/mapper.js";

const source = { ready: true, retries: 3 };
const result = mapObject(source, ["ready"] as const, value => Boolean(value));
if (result.ready !== true || Object.keys(result).length !== 1) {
  throw new Error("Single-key object mapping regression.");
}
if (source.retries !== 3) {
  throw new Error("Object mapper changed an unselected source property.");
}
