export function createCache() {
  return {
    get() { return undefined; },
    set() { return false; }
  };
}
