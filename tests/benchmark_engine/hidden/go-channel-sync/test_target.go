package main

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"time"

	"syntra-benchmark.local/channel-sync/syncutil"
)

func fail(message string) {
	fmt.Fprintln(os.Stderr, "SYNTRA_BENCHMARK_ASSERTION_FAILED:", message)
	os.Exit(1)
}

func main() {
	input := make(chan string, 6)
	for _, value := range []string{"Zulu", "alpha", "Zulu", "Beta", "alpha"} {
		input <- value
	}
	close(input)

	got := syncutil.CollectUnique(context.Background(), input, 4)
	want := []string{"Beta", "Zulu", "alpha"}
	if !reflect.DeepEqual(got, want) {
		fail(fmt.Sprintf("case-preserving concurrent uniqueness: got %v, want %v", got, want))
	}

	ctx, cancel := context.WithCancel(context.Background())
	openInput := make(chan string)
	done := make(chan []string, 1)
	go func() { done <- syncutil.CollectUnique(ctx, openInput, 3) }()
	cancel()
	select {
	case <-done:
	case <-time.After(750 * time.Millisecond):
		fail("workers did not stop after context cancellation")
	}

	empty := make(chan string)
	close(empty)
	if result := syncutil.CollectUnique(context.Background(), empty, 0); len(result) != 0 {
		fail("non-positive worker count must return an empty result")
	}
}
