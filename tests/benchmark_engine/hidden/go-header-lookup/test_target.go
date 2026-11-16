package main

import (
	"fmt"
	"os"

	"syntra-benchmark.local/header-lookup/headers"
)

func fail(message string) {
	fmt.Fprintln(os.Stderr, "SYNTRA_BENCHMARK_ASSERTION_FAILED:", message)
	os.Exit(1)
}

func main() {
	values := map[string]string{
		"X-Trace-ID": "upper-case-key",
		"x-trace-id": "lower-case-key",
		"Content-Type": "application/json",
	}
	if got := headers.Lookup(values, "x-trace-id", "missing"); got != "upper-case-key" {
		fail(fmt.Sprintf("case-insensitive deterministic lookup: got %q", got))
	}
	if got := headers.Lookup(values, "CONTENT-TYPE", "missing"); got != "application/json" {
		fail(fmt.Sprintf("case-insensitive header value: got %q", got))
	}
	if got := headers.Lookup(values, "unknown", "fallback"); got != "fallback" {
		fail("missing header must return the supplied fallback")
	}
	if got := headers.Lookup(nil, "x-trace-id", "fallback"); got != "fallback" {
		fail("nil maps must return fallback")
	}
	if got := headers.Lookup(values, "  ", "fallback"); got != "fallback" {
		fail("blank lookup names must return fallback")
	}
}
