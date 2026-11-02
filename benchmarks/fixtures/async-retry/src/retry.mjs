export async function retry(operation, options = {}) {
  return operation();
}
