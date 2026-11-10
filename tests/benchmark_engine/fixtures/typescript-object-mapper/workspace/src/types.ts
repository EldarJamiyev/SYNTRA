export type StringKeyOf<T> = Extract<keyof T, string>;

// The key bound is intentionally incomplete in this baseline fixture.
export type MappedRecord<TSource extends object, TKey extends string, TValue> = Record<TKey, TValue>;
