package hub

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// token is the Hugging Face access token, found as huggingface_hub finds
// it: $HF_TOKEN, else the file `hf auth login` writes ($HF_TOKEN_PATH, else
// $HF_HOME/token, else ~/.cache/huggingface/token). Empty when there is none.
var token = sync.OnceValue(func() string {
	if t := strings.TrimSpace(os.Getenv("HF_TOKEN")); t != "" {
		return t
	}
	path := os.Getenv("HF_TOKEN_PATH")
	if path == "" {
		home := os.Getenv("HF_HOME")
		if home == "" {
			if h, err := os.UserHomeDir(); err == nil {
				home = filepath.Join(h, ".cache", "huggingface")
			}
		}
		if home != "" {
			path = filepath.Join(home, "token")
		}
	}
	if path == "" {
		return ""
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
})
