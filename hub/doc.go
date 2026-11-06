// Package hub is a pure-Go client for the Hugging Face Hub download API. It
// uses net/http with Range-resume, caches to ~/.cache/huggingface/hub/, and
// supports both single-file downloads and full-snapshot fetches by glob.
package hub
