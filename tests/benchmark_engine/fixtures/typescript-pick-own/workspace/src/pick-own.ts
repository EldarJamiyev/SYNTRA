export function pickOwn<T extends object, K extends keyof T>(
  source: T,
  keys: readonly K[],
): Partial<Pick<T, K>> {
  const result = {} as Partial<Pick<T, K>>;
  for (const key of keys) {
    result[key] = source[key];
  }
  return result;
}
