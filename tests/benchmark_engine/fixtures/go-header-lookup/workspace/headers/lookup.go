package headers

func Lookup(values map[string]string, name, fallback string) string {
	if value, ok := values[name]; ok {
		return value
	}
	return fallback
}
