export function groupBy(items, keySelector) {
  if (!Array.isArray(items) || typeof keySelector !== "function") {
    throw new TypeError("items must be an array and keySelector must be callable");
  }
  const groups = {};
  for (let index = 0; index < items.length; index += 1) {
    const key = keySelector(items[index], index);
    if (typeof key !== "string") throw new TypeError("group keys must be strings");
    if (!Object.prototype.hasOwnProperty.call(groups, key)) groups[key] = [];
    groups[key].push(items[index]);
  }
  return groups;
}
