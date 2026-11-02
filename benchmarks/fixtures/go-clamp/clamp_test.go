package clamp

import "testing"

func TestClampBounds(t *testing.T) {
    if got := Clamp(-5, 0, 10); got != 0 { t.Fatalf("Clamp(-5) = %d, want 0", got) }
    if got := Clamp(15, 0, 10); got != 10 { t.Fatalf("Clamp(15) = %d, want 10", got) }
    if got := Clamp(4, 0, 10); got != 4 { t.Fatalf("Clamp(4) = %d, want 4", got) }
}
