import type { Result } from "./result.js";

export function mapResult<TValue, TNext, TError>(
  result: Result<TValue, TError>,
  mapper: (value: TValue) => TNext,
): Result<TNext, TError> {
  if (result.ok) {
    return { ok: true, value: mapper(result.value) };
  }
  return { ok: false, error: String(result.error) as unknown as TError };
}
