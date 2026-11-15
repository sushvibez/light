package hub

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"
)

// httpClient is used for every Hub request. It bounds connecting, the TLS
// handshake and the wait for response headers, but sets no overall
// deadline: checkpoint bodies can take hours. Callers cancel a stalled
// body through their context (DownloadContext and friends).
var httpClient = &http.Client{Transport: newTransport()}

func newTransport() http.RoundTripper {
	t := http.DefaultTransport.(*http.Transport).Clone()
	t.DialContext = (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	t.TLSHandshakeTimeout = 15 * time.Second
	t.ResponseHeaderTimeout = 60 * time.Second
	return t
}

// HTTPError is a Hub request that got a non-success status. Use errors.As
// to inspect the status of an error from this package:
//
//	var he *hub.HTTPError
//	if errors.As(err, &he) && he.Status == http.StatusNotFound { ... }
type HTTPError struct {
	Status int    // HTTP status code
	URL    string // requested URL (before redirects), or the local path for a checkpoint directory
	Body   string // start of the response body, for the message
}

func (e *HTTPError) Error() string {
	s := fmt.Sprintf("%s: HTTP %d", e.URL, e.Status)
	if e.Body != "" {
		s += ": " + e.Body
	}
	return s
}

// newHTTPError reads (a bounded prefix of) resp's body into an HTTPError
// for url.
func newHTTPError(url string, resp *http.Response) *HTTPError {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	return &HTTPError{Status: resp.StatusCode, URL: url, Body: string(body)}
}

// isHTTPNotFound reports whether err carries an HTTP 404 (from the Hub, or
// a file missing from a local checkpoint directory). It goes by the typed
// status, never the message, so a 503 whose body mentions "HTTP 404" is not
// mistaken for a missing file.
func isHTTPNotFound(err error) bool {
	var he *HTTPError
	return errors.As(err, &he) && he.Status == http.StatusNotFound
}
