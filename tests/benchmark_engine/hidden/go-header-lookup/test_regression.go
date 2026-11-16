package main

import (
	"fmt"
	"os"

	"syntra-benchmark.local/header-lookup/headers"
)

func main() {
	values := map[string]string{"Content-Type": "application/json"}
	if got := headers.Lookup(values, "Content-Type", "missing"); got != "application/json" {
		fmt.Fprintln(os.Stderr, "SYNTRA_BENCHMARK_ASSERTION_FAILED: exact-case lookup regression")
		os.Exit(1)
	}
	if got := headers.Lookup(values, "Accept", "none"); got != "none" {
		fmt.Fprintln(os.Stderr, "SYNTRA_BENCHMARK_ASSERTION_FAILED: fallback regression")
		os.Exit(1)
	}
}
