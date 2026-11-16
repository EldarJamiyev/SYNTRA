package retry

import "time"

func Backoff(attempt int, base, maximum time.Duration) time.Duration {
	if attempt < 1 || base <= 0 || maximum <= 0 {
		return 0
	}
	if base >= maximum {
		return maximum
	}
	return base
}
