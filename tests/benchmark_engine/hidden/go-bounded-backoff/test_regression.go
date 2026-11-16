package main

import (
	"fmt"
	"os"
	"time"

	"syntra-benchmark.local/bounded-backoff/retry"
)

func main() {
	if got := retry.Backoff(1, 250*time.Millisecond, time.Second); got != 250*time.Millisecond {
		fmt.Fprintln(os.Stderr, "SYNTRA_BENCHMARK_ASSERTION_FAILED: first retry must use the base delay")
		os.Exit(1)
	}
	if got := retry.Backoff(0, time.Second, time.Minute); got != 0 {
		fmt.Fprintln(os.Stderr, "SYNTRA_BENCHMARK_ASSERTION_FAILED: attempts are one-based")
		os.Exit(1)
	}
}
