package hub

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
	if revision == "" {
		revision = "main"
	}
	url := fmt.Sprintf("%s/api/models/%s/revision/%s?blobs=true", Endpoint(), repo, revision)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	setHeaders(req)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("hub.ListFiles: GET %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("hub.ListFiles: %s: HTTP %d: %s", repo, resp.StatusCode, string(body))
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
	if info.SHA != "" {
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
	return download(repo, revision, filename, progress)
}

// DownloadTo fetches repo's filename (at revision) to the path dst outside
// the Hub cache layout, resuming a partial dst+".part"; an existing
// non-empty dst is kept as is.
func DownloadTo(repo, revision, filename, dst string, progress Progress) error {
	if revision == "" {
		revision = "main"
	}
	if st, err := os.Stat(dst); err == nil && st.Size() > 0 {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	return fetchWith(fmt.Sprintf("%s/%s/resolve/%s/%s", Endpoint(), repo, revision, filename), dst, progress)
}

func setHeaders(req *http.Request) {
	req.Header.Set("User-Agent", "light/0.1 (+https://github.com/sushvibez/light)")
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
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
