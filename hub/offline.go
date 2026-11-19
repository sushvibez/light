package hub

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// ErrOffline is returned (wrapped) when HF_HUB_OFFLINE is set and what was
// asked for is not in the local cache. No request is made in offline mode.
var ErrOffline = errors.New("not in the local cache and HF_HUB_OFFLINE is set")

// Offline reports whether HF_HUB_OFFLINE is set to a true value ("1",
// "true", "yes", "on"; any case), as huggingface_hub reads it. Download,
// DownloadModelSafetensors, DownloadTo and ListFiles (and their Context
// variants) then serve only the cache: refs/<revision> → snapshots/<sha>,
// snapshots/<commit sha> for a SHA revision, or the legacy
// snapshots/<revision> layout.
func Offline() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("HF_HUB_OFFLINE"))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// listCached is ListFiles from the cache: the snapshot revision names
// (through refs/, or directly for a commit SHA), else the legacy
// snapshots/<revision> directory. Sizes are those of the cached files.
func listCached(repo, revision string) ([]RepoFile, error) {
	if err := checkRepoID(repo); err != nil {
		return nil, fmt.Errorf("hub.ListFiles: %w", err)
	}
	if err := checkRevision(revision); err != nil {
		return nil, fmt.Errorf("hub.ListFiles: %w", err)
	}
	cache, err := CacheDir()
	if err != nil {
		return nil, err
	}
	repoDir := filepath.Join(cache, "models--"+strings.ReplaceAll(repo, "/", "--"))
	var dirs []string
	if isCommitSHA(revision) {
		dirs = append(dirs, filepath.Join(repoDir, "snapshots", revision))
	} else if b, err := os.ReadFile(filepath.Join(repoDir, "refs", filepath.FromSlash(revision))); err == nil {
		if sha := strings.TrimSpace(string(b)); isCommitSHA(sha) {
			dirs = append(dirs, filepath.Join(repoDir, "snapshots", sha))
		}
	}
	dirs = append(dirs, filepath.Join(repoDir, "snapshots", filepath.FromSlash(revision)))
	for _, dir := range dirs {
		if st, err := os.Stat(dir); err != nil || !st.IsDir() {
			continue
		}
		var out []RepoFile
		err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
			if err != nil || d.IsDir() {
				return err
			}
			name := d.Name()
			if strings.HasSuffix(name, ".part") || strings.HasSuffix(name, ".part.url") || strings.HasSuffix(name, ".lock") {
				return nil // in-progress download state
			}
			st, err := os.Stat(p) // follows huggingface_hub's blob symlinks
			if err != nil || st.IsDir() {
				return nil
			}
			rel, err := filepath.Rel(dir, p)
			if err != nil {
				return err
			}
			out = append(out, RepoFile{Name: filepath.ToSlash(rel), Size: st.Size()})
			return nil
		})
		if err != nil {
			return nil, fmt.Errorf("hub.ListFiles: %w", err)
		}
		return out, nil
	}
	return nil, fmt.Errorf("hub.ListFiles: %s@%s: %w", repo, revision, ErrOffline)
}
