package main

import (
	"fmt"

	"github.com/michalswi/color"
)

func displayHelp() {
	fmt.Println(color.Format(color.GREEN, "Commands:"))
	fmt.Println(color.Format(color.GREEN, "  q  - quit: Exit the chat."))
	fmt.Println(color.Format(color.GREEN, "  r  - reset: Reset conversation history for both AIs and clear code block."))
	fmt.Println(color.Format(color.GREEN, "  h  - help: Display this help message."))
	fmt.Println(color.Format(color.GREEN, "  c  - code: Enter a code block. Specify a language (e.g., 'go', 'python') or press Enter for none, then enter code (end with '```' on a new line)."))
	fmt.Println(color.Format(color.GREEN, "  dc - display code block[s]: Display saved code block[s] for this session."))
	fmt.Println(color.Format(color.GREEN, "  Enter a message to join the AI-to-AI discussion or provide a prompt."))
	fmt.Println(color.Format(color.GREEN, "  Press Enter to continue the discussion without joining."))
}
