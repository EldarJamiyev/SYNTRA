package main

import (
	"context"
	"fmt"
	"os"
	"reflect"

	"syntra-benchmark.local/channel-sync/syncutil"
)

func main() {
	input := make(chan string, 3)
	input <- "same"
	input <- "same"
	close(input)
	got := syncutil.CollectUnique(context.Background(), input, 2)
	if !reflect.DeepEqual(got, []string{"same"}) {
		fmt.Fprintln(os.Stderr, "SYNTRA_BENCHMARK_ASSERTION_FAILED: duplicate input regression")
		os.Exit(1)
	}
}
