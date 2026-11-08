package syncutil

import (
	"context"
	"sort"
	"strings"
	"sync"
)

// CollectUnique consumes values concurrently and returns unique entries in stable order.
func CollectUnique(ctx context.Context, values <-chan string, workers int) []string {
	if workers < 1 {
		return nil
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	seen := make(map[string]struct{})
	result := make([]string, 0)

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for value := range values {
				key := strings.ToLower(value)
				mu.Lock()
				if _, exists := seen[key]; !exists {
					seen[key] = struct{}{}
					result = append(result, key)
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	sort.Strings(result)
	return result
}
