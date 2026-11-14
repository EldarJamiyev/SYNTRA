export async function mapConcurrent(items, mapper) {
  if (!Array.isArray(items) || typeof mapper !== "function") {
    throw new TypeError("items must be an array and mapper must be callable");
  }
  return Promise.all(items.map((item, index) => mapper(item, index)));
}
