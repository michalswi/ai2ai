package main

import (
	"fmt"

	"github.com/michalswi/color"
)

const version = "v0.1.1"

var banner = `
┏━┓╻┏━┓┏━┓╻
┣━┫┃┏━┛┣━┫┃
╹ ╹╹┗━╸╹ ╹╹
` + version + ` - @michalswi
`

func ShowBanner() {
	fmt.Printf("%s\n", color.Format(color.BLUE, banner))
}
