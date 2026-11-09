import { mapObject } from "../workspace/src/mapper.js";
import type { StringKeyOf } from "../workspace/src/types.js";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

type User = { id: number; name: string; enabled: boolean };
const user: User = { id: 17, name: "Ada", enabled: true };
const mapped = mapObject(user, ["id", "name"] as const, value => String(value));
const expectedShape: { id: string; name: string } = mapped;
const stringKey: StringKeyOf<User> = "enabled";
assert(stringKey === "enabled", "StringKeyOf must preserve valid string keys.");
assert(expectedShape.id === "17" && expectedShape.name === "Ada", "Selected keys must be mapped at runtime.");
assert(Object.keys(mapped).sort().join(",") === "id,name", "Unselected sibling keys must not leak into the result.");
assert(user.enabled === true, "Mapping must not mutate the source object.");

// @ts-expect-error A mapper must not accept keys that are absent from the source type.
mapObject(user, ["missing"] as const, value => String(value));
