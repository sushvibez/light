package hub

import (
	"fmt"
	"net/url"
	"path/filepath"
	"regexp"
	"strings"
)

// artifactPath validates a repository file name (a Hub "rfilename", or a
// shard name read from a checkpoint's index) and returns it as a relative
// OS path that stays inside whatever directory it is joined to.
//
// Hub names are slash-separated. Rejected: empty names or segments, "." and
// "..", rooted paths, backslashes (a Windows separator that would bypass the
// segment check), colons (drive letters, NTFS streams) and NUL, plus
// anything filepath.IsLocal refuses (e.g. reserved Windows device names).
func artifactPath(name string) (string, error) {
	if name == "" || strings.HasPrefix(name, "/") || strings.ContainsAny(name, "\\:\x00") {
		return "", fmt.Errorf("invalid repository path %q", name)
	}
	for _, seg := range strings.Split(name, "/") {
		if seg == "" || seg == "." || seg == ".." {
			return "", fmt.Errorf("invalid repository path %q", name)
		}
	}
	p := filepath.FromSlash(name)
	if !filepath.IsLocal(p) {
		return "", fmt.Errorf("invalid repository path %q", name)
	}
	return p, nil
}

// checkRevision validates a revision (branch, tag, "refs/pr/1", or commit
// SHA) before it becomes a cache path component (refs/<revision>, the
// legacy snapshots/<revision>). Slashes are allowed; traversal is not.
func checkRevision(rev string) error {
	if _, err := artifactPath(rev); err != nil {
		return fmt.Errorf("invalid revision %q", rev)
	}
	return nil
}

var repoIDRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?$`)

// checkRepoID validates a Hub repository id ("org/name" or "name") before it
// becomes the models--org--name cache directory.
func checkRepoID(repo string) error {
	if !repoIDRe.MatchString(repo) || strings.Contains(repo, "..") {
		return fmt.Errorf("invalid repository id %q", repo)
	}
	return nil
}

// isCommitSHA reports whether s is a full lowercase hex git commit id
// (SHA-1, or SHA-256 for repositories using it): the only form accepted as
// a snapshots/<sha> directory name.
func isCommitSHA(s string) bool {
	if len(s) != 40 && len(s) != 64 {
		return false
	}
	for i := 0; i < len(s); i++ {
		if c := s[i]; (c < '0' || c > '9') && (c < 'a' || c > 'f') {
			return false
		}
	}
	return true
}

// resolveURL is <endpoint>/<repo>/resolve/<revision>/<filename>, with the
// revision escaped as one path segment ("refs/pr/1" → refs%2Fpr%2F1, as
// huggingface_hub sends it) and each filename segment escaped.
func resolveURL(repo, revision, filename string) string {
	segs := strings.Split(filename, "/")
	for i, s := range segs {
		segs[i] = url.PathEscape(s)
	}
	return fmt.Sprintf("%s/%s/resolve/%s/%s", Endpoint(), repo, url.PathEscape(revision), strings.Join(segs, "/"))
}
