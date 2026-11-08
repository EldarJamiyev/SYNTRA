import type { MappedRecord } from "./types.js";

export function mapObject<TSource extends object, TKey extends string, TValue>(
  source: TSource,
  keys: readonly TKey[],
  mapper: (value: unknown, key: TKey) => TValue
): MappedRecord<TSource, TKey, TValue> {
  const output = {} as MappedRecord<TSource, TKey, TValue>;
  for (const key of keys) {
    output[key] = mapper(source[key as keyof TSource], key);
  }
  return output;
}
