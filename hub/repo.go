package hub

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
)

// RepoFile is one file of a Hub repository.
type RepoFile struct {
	Name string // path inside the repo ("model-00001-of-00002.safetensors")
	Size int64  // bytes (0 when the Hub does not report it)
}

// ListFiles returns the files of repo at revision (default "main"), with
// sizes, from the Hub model API.
func ListFiles(repo, revision string) ([]RepoFile, error) {
	return ListFilesContext(context.Background(), repo, revision)
}

// ListFilesContext is ListFiles with a context for canceling the request.
func ListFilesContext(ctx context.Context, repo, revision string) ([]RepoFile, error) {
	if revision == "" {
		revision = "main"
	}
	apiURL := fmt.Sprintf("%s/api/models/%s/revision/%s?blobs=true", Endpoint(), repo, url.PathEscape(revision))
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, apiURL, nil)
	if err != nil {
		return nil, err
	}
	setHeaders(req)
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("hub.ListFiles: GET %s: %w", apiURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("hub.ListFiles: %s: GET %w", repo, newHTTPError(apiURL, resp))
	}
	var info struct {
		SHA      string `json:"sha"`
		Siblings []struct {
			Name string `json:"rfilename"`
			Size int64  `json:"size"`
			LFS  *struct {
				Size int64 `json:"size"`
			} `json:"lfs"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("hub.ListFiles: parse: %w", err)
	}
	if isCommitSHA(info.SHA) {
		shaCacheMu.Lock()
		shaCache[repo+"@"+revision] = info.SHA
		shaCacheMu.Unlock()
	}
	out := make([]RepoFile, 0, len(info.Siblings))
	for _, s := range info.Siblings {
		f := RepoFile{Name: s.Name, Size: s.Size}
		if f.Size == 0 && s.LFS != nil {
			f.Size = s.LFS.Size
		}
		out = append(out, f)
	}
	return out, nil
}

// Progress reports a download's bytes so far and its total (0 if unknown).
type Progress func(done, total int64)

// DownloadWithProgress is Download reporting progress while it fetches (a
// file already in the cache returns at once without calling progress).
func DownloadWithProgress(repo, revision, filename string, progress Progress) (string, error) {
	return DownloadWithProgressContext(context.Background(), repo, revision, filename, progress)
}

// DownloadWithProgressContext is DownloadWithProgress with a context (see
// DownloadContext).
func DownloadWithProgressContext(ctx context.Context, repo, revision, filename string, progress Progress) (string, error) {
	p, _, err := downloadAt(ctx, repo, revision, filename, progress)
	return p, err
}

// DownloadTo fetches repo's filename (at revision) to the path dst outside
// the Hub cache layout, resuming a partial dst+".part"; an existing
// non-empty dst is kept as is. dst is the caller's explicit choice and is
// used as given; filename must be a valid repository path. The bytes come
// from the commit revision resolves to (falling back to the revision itself
// when the API is unreachable). A .part is resumed only when it provably
// belongs to the same object (see fetchWith), so it never mixes commits.
func DownloadTo(repo, revision, filename, dst string, progress Progress) error {
	return DownloadToContext(context.Background(), repo, revision, filename, dst, progress)
}

// DownloadToContext is DownloadTo with a context (see DownloadContext).
func DownloadToContext(ctx context.Context, repo, revision, filename, dst string, progress Progress) error {
	if revision == "" {
		revision = "main"
	}
	if _, err := artifactPath(filename); err != nil {
		return fmt.Errorf("hub.DownloadTo: %w", err)
	}
	if st, err := os.Stat(dst); err == nil && st.Size() > 0 {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	rev := revision
	if !isCommitSHA(rev) {
		if sha, err := resolveCommitSHA(ctx, repo, revision); err == nil {
			rev = sha
		} else if cerr := ctx.Err(); cerr != nil {
			return fmt.Errorf("hub.DownloadTo: %w", cerr)
		}
	}
	return fetchLocked(ctx, resolveURL(repo, rev, filename), dst, isCommitSHA(rev), progress)
}

func setHeaders(req *http.Request) {
	req.Header.Set("User-Agent", "light/0.1 (+https://github.com/sushvibez/light)")
	if tok := token(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
}

// progressWriter counts bytes written and reports them.
type progressWriter struct {
	w           io.Writer
	done, total int64
	report      Progress
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)
	p.report(p.done, p.total)
	return n, err
}
