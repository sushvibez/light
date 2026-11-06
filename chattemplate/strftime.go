package chattemplate

import (
	"fmt"
	"strings"
	"time"
)

// Strftime formats t like Python's datetime.strftime for the directives
// chat templates use (%d %b %B %Y %m %y %H %M %S %I %p %a %A %j %f %w %e
// %%, plus glibc's %-d style no-padding). Unknown directives are an error.
func Strftime(t time.Time, format string) (string, error) {
	var b strings.Builder
	for i := 0; i < len(format); i++ {
		c := format[i]
		if c != '%' {
			b.WriteByte(c)
			continue
		}
		if i+1 >= len(format) {
			return "", fmt.Errorf("strftime: trailing %%")
		}
		i++
		pad := true
		if format[i] == '-' && i+1 < len(format) {
			pad = false
			i++
		}
		num := func(v, width int) {
			if pad {
				fmt.Fprintf(&b, "%0*d", width, v)
			} else {
				fmt.Fprintf(&b, "%d", v)
			}
		}
		hour12 := t.Hour() % 12
		if hour12 == 0 {
			hour12 = 12
		}
		switch format[i] {
		case 'd':
			num(t.Day(), 2)
		case 'e':
			fmt.Fprintf(&b, "%2d", t.Day())
		case 'm':
			num(int(t.Month()), 2)
		case 'y':
			num(t.Year()%100, 2)
		case 'Y':
			fmt.Fprintf(&b, "%d", t.Year())
		case 'H':
			num(t.Hour(), 2)
		case 'I':
			num(hour12, 2)
		case 'M':
			num(t.Minute(), 2)
		case 'S':
			num(t.Second(), 2)
		case 'j':
			num(t.YearDay(), 3)
		case 'f':
			fmt.Fprintf(&b, "%06d", t.Nanosecond()/1000)
		case 'w':
			fmt.Fprintf(&b, "%d", int(t.Weekday()))
		case 'p':
			if t.Hour() < 12 {
				b.WriteString("AM")
			} else {
				b.WriteString("PM")
			}
		case 'a':
			b.WriteString(t.Weekday().String()[:3])
		case 'A':
			b.WriteString(t.Weekday().String())
		case 'b', 'h':
			b.WriteString(t.Month().String()[:3])
		case 'B':
			b.WriteString(t.Month().String())
		case '%':
			b.WriteByte('%')
		default:
			return "", fmt.Errorf("strftime: unsupported directive %%%c", format[i])
		}
	}
	return b.String(), nil
}
