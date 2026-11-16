package main

import (
	"fmt"
	"os"
	"time"

	"syntra-benchmark.local/bounded-backoff/retry"
)

func fail(message string) {
	fmt.Fprintln(os.Stderr, "SYNTRA_BENCHMARK_ASSERTION_FAILED:", message)
	os.Exit(1)
}

func main() {
	if got := retry.Backoff(1, 100*time.Millisecond, time.Second); got != 100*time.Millisecond {
		fail(fmt.Sprintf("attempt one: got %v", got))
	}
	if got := retry.Backoff(3, 100*time.Millisecond, time.Second); got != 400*time.Millisecond {
		fail(fmt.Sprintf("exponential step: got %v, want 400ms", got))
	}
	if got := retry.Backoff(8, 100*time.Millisecond, 500*time.Millisecond); got != 500*time.Millisecond {
		fail(fmt.Sprintf("upper cap: got %v, want 500ms", got))
	}
	if got := retry.Backoff(int(^uint(0)>>1), 100*time.Millisecond, 500*time.Millisecond); got != 500*time.Millisecond {
		fail(fmt.Sprintf("very large attempt must saturate safely: got %v", got))
	}
	for _, got := range []time.Duration{
		retry.Backoff(0, time.Second, time.Minute),
		retry.Backoff(1, 0, time.Minute),
		retry.Backoff(1, time.Second, 0),
		retry.Backoff(1, -time.Second, time.Minute),
	} {
		if got != 0 {
			fail(fmt.Sprintf("invalid bounds/attempt must return zero, got %v", got))
		}
	}
}
