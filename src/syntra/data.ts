import { tokenizer } from './tokenizer';

export interface CorpusFile {
  id: string;
  name: string;
  language: string;
  content: string;
  size: number;
}

export const DEFAULT_CORPUS_FILES: CorpusFile[] = [
  {
    id: 'f1',
    name: 'algorithms.py',
    language: 'python',
    content: `def fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    sequence = [0, 1]
    while len(sequence) < n:
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

def binary_search(array: list[int], target: int) -> int:
    """Find target in sorted array, returns index or -1."""
    left = 0
    right = len(array) - 1
    while left <= right:
        mid = (left + right) // 2
        if array[mid] == target:
            return mid
        elif array[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def quicksort(items: list[int]) -> list[int]:
    """Standard in-place or recursion quicksort."""
    if len(items) <= 1:
        return items
    pivot = items[len(items) // 2]
    left = [x for x in items if x < pivot]
    middle = [x for x in items if x == pivot]
    right = [x for x in items if x > pivot]
    return quicksort(left) + middle + quicksort(right)
`,
    size: 928,
  },
  {
    id: 'f2',
    name: 'data_structures.py',
    language: 'python',
    content: `class Queue:
    """FIFO queue implementation."""
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError("dequeue from empty queue")

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def size(self) -> int:
        return len(self.items)

class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        if not self.head:
            self.head = Node(val)
            return
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = Node(val)
`,
    size: 785,
  },
  {
    id: 'f3',
    name: 'server.py',
    language: 'python',
    content: `from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {"status": "ok", "service": "SYNTRA Local Runtime"}
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

def run(host: str = "127.0.0.1", port: int = 8000):
    server = HTTPServer((host, port), HealthHandler)
    print(f"Server started on http://{host}:{port}")
    server.serve_forever()
`,
    size: 642,
  },
  {
    id: 'f4',
    name: 'utils.ts',
    language: 'typescript',
    content: `export function clamp(val: number, min: number, max: number): number {
  return Math.min(Math.max(val, min), max);
}

export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}
`,
    size: 340,
  },
];

export interface PreparationResult {
  files: number;
  skipped: number;
  source_bytes: number;
  tokens: number;
  output: string;
}

export function prepareCorpusFromFiles(files: CorpusFile[]): {
  result: PreparationResult;
  tokenStream: Uint16Array;
} {
  const tokenList: number[] = [];
  let sourceBytes = 0;
  let fileCount = 0;
  let skipped = 0;

  for (const file of files) {
    if (!file.content || file.content.trim().length === 0) {
      skipped++;
      continue;
    }
    const fileBytes = new TextEncoder().encode(file.content).length;
    sourceBytes += fileBytes;
    fileCount++;

    const encoded = tokenizer.encode(file.content, { addBos: true, addEos: true });
    for (let i = 0; i < encoded.length; i++) {
      tokenList.push(encoded[i]);
    }
    // Newline token ID (ASCII 10 + BYTE_OFFSET 4 = 14) separates adjacent files
    tokenList.push(14);
  }

  if (fileCount === 0) {
    throw new Error('No usable source files were provided in raw data. Add code files first.');
  }

  const tokenStream = new Uint16Array(tokenList);
  return {
    result: {
      files: fileCount,
      skipped,
      source_bytes: sourceBytes,
      tokens: tokenStream.length,
      output: 'data/processed/tokens.bin',
    },
    tokenStream,
  };
}
