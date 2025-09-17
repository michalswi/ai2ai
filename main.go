package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/michalswi/color"
	openai "github.com/sashabaranov/go-openai"
)

const (
	defaultAITurns = "3"
	appName        = "ai2ai"
	defaultModel   = openai.GPT5Mini
	// https://docs.x.ai/docs/api-reference#chat-completions
	defaultEndpoint  = "https://api.x.ai/v1/chat/completions"
	defaultGrokModel = "grok-4"

	requestContentInit           = "You are a helpful assistant that can answer questions and help with tasks. Do not prefix your response with your name."
	requestContentDiscussionUser = "You are discussing with another AI and the user. Respond to the latest points, including user input, and provide your perspective or answer. Do not prefix your response with your name."
	requestContentDiscussionAuto = "You are discussing with another AI without user. Respond to the latest points, including user input, and provide your perspective or answer. Do not prefix your response with your name."
	requestDiscussionPrompt      = "Review the discussion, respond to the latest points (including any user input), and provide your perspective or answer. Do not prefix your response with your name."
)

// Session represents a chat session to be saved
type Session struct {
	Timestamp       string           `json:"timestamp"`
	UserInputs      []string         `json:"user_inputs"`
	CodeBlocks      []CodeBlock      `json:"code_blocks"`
	AIConversations []AIResponse     `json:"ai_conversations"`
	DiscussionTurns []DiscussionTurn `json:"discussion_turns"`
}

// CodeBlock represents a code block entered by the user
type CodeBlock struct {
	Language string `json:"language"`
	Code     string `json:"code"`
}

// AIResponse represents a single AI response
type AIResponse struct {
	AIName   string `json:"ai_name"`
	Response string `json:"response"`
}

// DiscussionTurn represents a single turn in the AI-to-AI discussion
type DiscussionTurn struct {
	TurnNumber int          `json:"turn_number"`
	Responses  []AIResponse `json:"responses"`
}

// GrokClient represents a client for interacting with xAI's Grok API
type GrokClient struct {
	apiKey       string
	endpoint     string
	httpClient   *http.Client
	conversation []Message // Maintain conversation history
}

// NewGrokClient creates a new Grok client with the provided API key
func NewGrokClient(apiKey, endpoint string) *GrokClient {
	return &GrokClient{
		apiKey:     apiKey,
		endpoint:   endpoint,
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}
}

// Message represents a single message in the chat API request
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// APIRequest represents the request body for the chat completions API
type APIRequest struct {
	Messages    []Message `json:"messages"`
	Model       string    `json:"model"`
	Stream      bool      `json:"stream"`
	Temperature float64   `json:"temperature"`
}

// APIResponse represents the structure of the chat completions API response
type APIResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int     `json:"index"`
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// AIClient interface defines the contract for AI clients
type AIClient interface {
	GenerateResponse(ctx context.Context, prompt string, aiModel string) (string, error)
	GenerateDiscussionResponse(ctx context.Context, userPrompt, discussionHistory, aiModel string, auto bool) (string, error)
	Name(aiModel string) string
	ResetConversation()
}

// Adapter for OpenAI client to match AIClient interface
type OpenAIClientAdapter struct {
	client       *openai.Client
	conversation []openai.ChatCompletionMessage
}

// saveSessionToFile saves the provided Session as a timestamped JSON file in /tmp.
// Returns an error if JSON marshaling or file write fails.
func saveSessionToFile(session Session) error {
	// Generate timestamp-based filename (e.g., ai2ai_20250820_0847.json)
	timestamp := time.Now().UTC().Format("20060102_1504")
	filename := filepath.Join("/tmp", fmt.Sprintf("ai2ai_%s.json", timestamp))

	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal session to JSON: %w", err)
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write session to %s: %w", filename, err)
	}

	log.Printf("Session saved to %s", filename)
	return nil
}

// NewOpenAIClientAdapter creates and returns an OpenAIClientAdapter configured with the provided API key.
func NewOpenAIClientAdapter(apiKey string) *OpenAIClientAdapter {
	return &OpenAIClientAdapter{
		client: openai.NewClient(apiKey),
	}
}

// Name returns a human-readable name for the OpenAI adapter including the model.
func (a *OpenAIClientAdapter) Name(aiModel string) string {
	name := fmt.Sprintf("OpenAI (%s)", aiModel)
	return name
}

// sendRequest sends the provided messages to OpenAI's chat completions API, retries on transient errors,
// and returns the assistant's reply (or an error). Context is honored for timeouts/cancellation.
func (a *OpenAIClientAdapter) sendRequest(ctx context.Context, aiModel string, messages []openai.ChatCompletionMessage) (string, error) {
	// todo - debug
	// fmt.Println(messages)

	var resp openai.ChatCompletionResponse
	var err error
	for retries := 0; retries < 3; retries++ {
		resp, err = a.client.CreateChatCompletion(
			ctx,
			openai.ChatCompletionRequest{
				Model:    aiModel,
				Messages: messages,
			},
		)
		if err == nil {
			break
		}
		// retry only on transient errors
		if strings.Contains(err.Error(), "context canceled") || strings.Contains(err.Error(), "rate limit") {
			time.Sleep(time.Second * time.Duration(retries+1))
			continue
		}
		break
	}
	if err != nil {
		return "", fmt.Errorf("OpenAI request failed: %v", err)
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no choices in OpenAI response")
	}

	answer := strings.TrimSpace(resp.Choices[0].Message.Content)
	if answer == "" {
		return "", fmt.Errorf("empty response from OpenAI")
	}
	return answer, nil
}

// GenerateResponse appends the required system and user messages, sends the request,
// appends the assistant reply to the conversation history, and returns the reply.
func (a *OpenAIClientAdapter) GenerateResponse(ctx context.Context, prompt string, aiModel string) (string, error) {
	a.conversation = append(a.conversation, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: requestContentInit,
	}, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: prompt,
	})

	answer, err := a.sendRequest(ctx, aiModel, a.conversation)
	if err != nil {
		return "", err
	}

	a.conversation = append(a.conversation, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: answer,
	})
	return answer, nil
}

// GenerateDiscussionResponse prepares a discussion prompt (system + user), sends it to OpenAI model and appends
// the assistant's reply to the conversation history. The 'auto' flag selects an alternate system prompt.
func (a *OpenAIClientAdapter) GenerateDiscussionResponse(ctx context.Context, userPrompt, discussionHistory, aiModel string, auto bool) (string, error) {
	discussionPrompt := fmt.Sprintf("The user asked: '%s'. The discussion so far (including user input): '%s'. %s", userPrompt, discussionHistory, requestDiscussionPrompt)

	reqContent := requestContentDiscussionUser
	if auto {
		reqContent = requestContentDiscussionAuto
	}

	a.conversation = append(a.conversation, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleSystem,
		Content: reqContent,
	}, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleUser,
		Content: discussionPrompt,
	})

	answer, err := a.sendRequest(ctx, aiModel, a.conversation)
	if err != nil {
		return "", err
	}

	a.conversation = append(a.conversation, openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: answer,
	})
	return answer, nil
}

func (a *OpenAIClientAdapter) ResetConversation() {
	a.conversation = nil
}

// Name returns a human-readable name for the Grok adapter including the model.
func (c *GrokClient) Name(aiModel string) string {
	name := fmt.Sprintf("Grok (%s)", aiModel)
	return name
}

// sendRequest sends the provided messages to Grok's chat completions API, retries on transient errors,
// and returns the assistant's reply (or an error). Context is honored for timeouts/cancellation.
func (c *GrokClient) sendRequest(ctx context.Context, reqBody APIRequest, aiModel string) (string, error) {
	// todo - debug
	// fmt.Println(reqBody)

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request body: %v", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.endpoint, bytes.NewBuffer(jsonBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	var resp *http.Response
	for retries := 0; retries < 3; retries++ {
		resp, err = c.httpClient.Do(req)
		if err == nil || (!strings.Contains(err.Error(), "context canceled") && !strings.Contains(err.Error(), "rate limit")) {
			break
		}
		time.Sleep(time.Second * time.Duration(retries+1))
	}
	if err != nil {
		return "", fmt.Errorf("failed to send request after retries: %v", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %v", err)
	}

	var apiResp APIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", fmt.Errorf("failed to parse response: %v", err)
	}

	if len(apiResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	response := strings.TrimSpace(apiResp.Choices[0].Message.Content)
	if response == "" {
		return "", fmt.Errorf("empty response from API")
	}

	c.conversation = append(c.conversation, Message{
		Role:    "assistant",
		Content: response,
	})

	if strings.Contains(strings.ToLower(response), "grok-1.5") && strings.Contains(apiResp.Model, aiModel) {
		log.Printf("Warning: Grok response claims Grok-1.5, but API metadata indicates %s. This may be a bug.", apiResp.Model)
	}

	return response, nil
}

// GenerateResponse appends the required system and user messages, sends the request,
// appends the assistant reply to the conversation history, and returns the reply.
func (c *GrokClient) GenerateResponse(ctx context.Context, prompt string, aiModel string) (string, error) {
	c.conversation = append(c.conversation, Message{
		Role:    "system",
		Content: requestContentInit,
	}, Message{
		Role:    "user",
		Content: prompt,
	})

	reqBody := APIRequest{
		Messages:    c.conversation,
		Model:       aiModel,
		Stream:      false,
		Temperature: 0,
	}

	return c.sendRequest(ctx, reqBody, aiModel)
}

// GenerateDiscussionResponse prepares a discussion prompt (system + user), sends it to Grok model and appends
// the assistant's reply to the conversation history. The 'auto' flag selects an alternate system prompt.
func (c *GrokClient) GenerateDiscussionResponse(ctx context.Context, userPrompt, discussionHistory, aiModel string, auto bool) (string, error) {
	discussionPrompt := fmt.Sprintf("The user asked: '%s'. The discussion so far (including user input): '%s'. %s", userPrompt, discussionHistory, requestDiscussionPrompt)

	reqContent := requestContentDiscussionUser
	if auto {
		reqContent = requestContentDiscussionAuto
	}

	c.conversation = append(c.conversation, Message{
		Role:    "system",
		Content: reqContent,
	}, Message{
		Role:    "user",
		Content: discussionPrompt,
	})

	reqBody := APIRequest{
		Messages:    c.conversation,
		Model:       aiModel,
		Stream:      false,
		Temperature: 0,
	}

	return c.sendRequest(ctx, reqBody, aiModel)
}

func (c *GrokClient) ResetConversation() {
	c.conversation = nil
}

// cleanResponse removes any leading AI name prefix from the response
func cleanResponse(response, aiName string) string {
	prefix := aiName + ":"
	response = strings.TrimSpace(response)
	if strings.HasPrefix(response, prefix) {
		return strings.TrimSpace(response[len(prefix):])
	}
	return response
}

// ParallelUserAIChatWithAgreement runs an interactive chat loop coordinating a human user with two AI clients.
func ParallelUserAIChatWithAgreement(ai1, ai2 AIClient, openaiModel, grokModel string, maxAITurns int) {
	fmt.Printf("%s Interactive chat with %s and %s started.\nNumber of discussion turns is %d. \nCommands: 'q' to quit, 'r' to reset history, 'h' for help, 'c' to enter code.\n",
		color.Format(color.GREEN, "Starting"), ai1.Name(openaiModel), ai2.Name(grokModel), maxAITurns)

	session := Session{
		Timestamp:       time.Now().UTC().Format(time.RFC3339),
		UserInputs:      []string{},
		CodeBlocks:      []CodeBlock{},
		AIConversations: []AIResponse{},
		DiscussionTurns: []DiscussionTurn{},
	}

	reader := bufio.NewReader(os.Stdin)
	var userIntervention string
	var codeBlock string    // Store the current code block
	var codeLanguage string // Store the language for the code block

	defer func() {
		// Save session when the function exits (e.g., on quit or error)
		if err := saveSessionToFile(session); err != nil {
			log.Printf("Error saving session: %v", err)
		}
	}()

	for {
		fmt.Printf("%s You (type initial prompt or command): ", color.Format(color.YELLOW, time.Now().UTC().Format(time.RFC1123)))
		userInput, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("%s Error reading input: %v\n", color.Format(color.RED, "Error"), err)
			break
		}
		userInput = strings.TrimSpace(userInput)

		if userInput == "q" {
			fmt.Printf("%s Chat stopped.\n", color.Format(color.GREEN, "Exiting"))
			break
		}
		if userInput == "h" {
			displayHelp()
			continue
		}
		if userInput == "r" {
			ai1.ResetConversation()
			ai2.ResetConversation()
			codeBlock = ""
			codeLanguage = ""
			session.UserInputs = []string{}
			session.CodeBlocks = []CodeBlock{}
			session.AIConversations = []AIResponse{}
			session.DiscussionTurns = []DiscussionTurn{}
			fmt.Printf("%s Conversation history reset for both AIs.\n", color.Format(color.GREEN, "Reset"))
			continue
		}
		if userInput == "c" {
			fmt.Printf("%s Enter programming language (e.g., 'go', 'python', or press Enter for 'none'):\n", color.Format(color.YELLOW, "Code Input"))
			language, err := reader.ReadString('\n')
			if err != nil {
				fmt.Printf("%s Error reading language input: %v\n", color.Format(color.RED, "Error"), err)
				continue
			}
			codeLanguage = strings.TrimSpace(language)

			fmt.Printf("%s Enter code block (end with '```' on a new line):\n", color.Format(color.YELLOW, "Code Input"))
			codeLines := []string{}
			for {
				line, err := reader.ReadString('\n')
				if err != nil {
					fmt.Printf("%s Error reading code input: %v\n", color.Format(color.RED, "Error"), err)
					break
				}
				line = strings.TrimRight(line, "\r\n")
				if line == "```" {
					break
				}
				codeLines = append(codeLines, line)
			}
			codeBlock = strings.Join(codeLines, "\n")
			if codeBlock == "" {
				fmt.Printf("%s No code entered. Returning to main prompt.\n", color.Format(color.YELLOW, "Note"))
				codeLanguage = ""
				continue
			}

			if codeLanguage == "" || strings.EqualFold(codeLanguage, "none") {
				codeLanguage = "nolanguage"
			}

			// Save code block to session
			session.CodeBlocks = append(session.CodeBlocks, CodeBlock{
				Language: codeLanguage,
				Code:     codeBlock,
			})
			fmt.Printf("%s Code block saved (language: %s)\n", color.Format(color.GREEN, "Code Saved"),
				codeLanguage)
			continue
			// fmt.Printf("%s Code block saved (language: %s):\n%s\n", color.Format(color.GREEN, "Code Saved"),
			// 	codeLanguage, codeBlock)
			// continue
		}

		if userInput == "dc" {
			displayCodeBlocks(session)
			continue
		}

		if userInput == "" {
			continue
		}

		// Save user input to session
		session.UserInputs = append(session.UserInputs, userInput)

		// Combine user input with code block if available
		prompt := userInput
		if codeBlock != "" {
			if codeLanguage != "" {
				prompt = fmt.Sprintf("%s\n\n[Code Context]:\n```%s\n%s\n```", userInput, codeLanguage, codeBlock)
			} else {
				prompt = fmt.Sprintf("%s\n\n[Code Context]:\n```\n%s\n```", userInput, codeBlock)
			}
		}

		// Phase 1: Both AIs respond to the user prompt concurrently
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		var wg sync.WaitGroup
		type aiResponse struct {
			name     string
			response string
			err      error
		}
		responses := make(chan aiResponse, 2)

		// Launch goroutine for AI1 (OpenAI)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fmt.Printf("%s Waiting for %s...\n", color.Format(color.YELLOW, "Processing"), ai1.Name(openaiModel))
			resp, err := ai1.GenerateResponse(ctx, prompt, openaiModel)
			responses <- aiResponse{name: ai1.Name(openaiModel), response: resp, err: err}
		}()

		// Launch goroutine for AI2 (Grok)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fmt.Printf("%s Waiting for %s...\n", color.Format(color.YELLOW, "Processing"), ai2.Name(grokModel))
			resp, err := ai2.GenerateResponse(ctx, prompt, grokModel)
			responses <- aiResponse{name: ai2.Name(grokModel), response: resp, err: err}
		}()

		// Wait for both responses and close channel
		go func() {
			wg.Wait()
			close(responses)
		}()

		// Collect initial responses
		ai1Response, ai2Response := "", ""
		ai1Name, ai2Name := ai1.Name(openaiModel), ai2.Name(grokModel)
		for resp := range responses {
			if resp.err != nil {
				fmt.Printf("%s Error from %s: %v\n", color.Format(color.RED, "Error"), resp.name, resp.err)
				continue
			}
			// Clean the response to remove any AI name prefix
			cleanedResp := cleanResponse(resp.response, resp.name)
			fmt.Printf("%s %s\n", color.Format(color.GREEN, resp.name), cleanedResp)
			fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))

			if resp.name == ai1Name {
				ai1Response = cleanedResp
			} else if resp.name == ai2Name {
				ai2Response = cleanedResp
			}
		}
		// Cancel context after both responses are collected
		cancel()

		// Save initial responses to session.AIConversations
		if ai1Response != "" {
			session.AIConversations = append(session.AIConversations, AIResponse{
				AIName:   ai1Name,
				Response: ai1Response,
			})
		}
		if ai2Response != "" {
			session.AIConversations = append(session.AIConversations, AIResponse{
				AIName:   ai2Name,
				Response: ai2Response,
			})
		}

		// Skip discussion if either response failed
		if ai1Response == "" || ai2Response == "" {
			fmt.Printf("%s Discussion skipped due to missing responses.\n", color.Format(color.YELLOW, "Note"))
			continue
		}

		// Prompt user to start discussion
		fmt.Printf("(Enter to continue, or type 'c' to add code, 'dc' to display code blocks, 'q' to quit, 'h' for help, 'r' to reset): ")
		userIntervention, err = reader.ReadString('\n')
		if err != nil {
			fmt.Printf("%s Error reading input: %v\n", color.Format(color.RED, "Error"), err)
			break
		}
		userIntervention = strings.TrimSpace(userIntervention)
		if userIntervention == "q" {
			fmt.Printf("%s Chat stopped.\n", color.Format(color.GREEN, "Exiting"))
			break
		}
		if userIntervention == "h" {
			displayHelp()
			continue
		}
		if userIntervention == "r" {
			ai1.ResetConversation()
			ai2.ResetConversation()
			codeBlock = ""
			codeLanguage = ""
			session.UserInputs = []string{}
			session.CodeBlocks = []CodeBlock{}
			session.AIConversations = []AIResponse{}
			session.DiscussionTurns = []DiscussionTurn{}
			fmt.Printf("%s Conversation history reset for both AIs.\n", color.Format(color.GREEN, "Reset"))
			continue
		}
		if userIntervention == "c" {
			fmt.Printf("%s Enter programming language (e.g., 'go', 'python', or press Enter for 'none'):\n", color.Format(color.YELLOW, "Code Input"))
			language, err := reader.ReadString('\n')
			if err != nil {
				fmt.Printf("%s Error reading language input: %v\n", color.Format(color.RED, "Error"), err)
				continue
			}
			codeLanguage = strings.TrimSpace(language)

			fmt.Printf("%s Enter code block (end with '```' on a new line):\n", color.Format(color.YELLOW, "Code Input"))
			codeLines := []string{}
			for {
				line, err := reader.ReadString('\n')
				if err != nil {
					fmt.Printf("%s Error reading code input: %v\n", color.Format(color.RED, "Error"), err)
					break
				}
				line = strings.TrimRight(line, "\r\n")
				if line == "```" {
					break
				}
				codeLines = append(codeLines, line)
			}
			codeBlock = strings.Join(codeLines, "\n")
			if codeBlock == "" {
				fmt.Printf("%s No code entered. Returning to main prompt.\n", color.Format(color.YELLOW, "Note"))
				codeLanguage = ""
				continue
			}

			if codeLanguage == "" || strings.EqualFold(codeLanguage, "none") {
				codeLanguage = "nolanguage"
			}

			// Save code block to session
			session.CodeBlocks = append(session.CodeBlocks, CodeBlock{
				Language: codeLanguage,
				Code:     codeBlock,
			})
			fmt.Printf("%s Code block saved (language: %s)\n", color.Format(color.GREEN, "Code Saved"), codeLanguage)
			continue
		}
		if userIntervention == "dc" {
			displayCodeBlocks(session)
			continue
		}
		if userIntervention != "" {
			// User joins discussion early
			fmt.Printf("%s User (you): %s\n", color.Format(color.BLUE, "User"), userIntervention)
		}

		// Phase 2: AIs discuss with user participation, OpenAI first, then Grok
		fmt.Printf("%s Starting AI-to-AI discussion...\n", color.Format(color.YELLOW, "Discussion"))
		fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
		discussionHistory := fmt.Sprintf("%s: %s\n%s: %s", ai1Name, ai1Response, ai2Name, ai2Response)
		if codeBlock != "" {
			if codeLanguage != "" {
				discussionHistory = fmt.Sprintf("[Code Context]:\n```%s\n%s\n```\n\n%s", codeLanguage, codeBlock, discussionHistory)
			} else {
				discussionHistory = fmt.Sprintf("[Code Context]:\n```\n%s\n```\n\n%s", codeBlock, discussionHistory)
			}
		}
		if userIntervention != "" {
			discussionHistory += fmt.Sprintf("\nUser: %s", userIntervention)
		}

		for turn := 0; turn < maxAITurns; turn++ {
			// '30*time.Second' causes 'context deadline exceeded'
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			discussionTurn := DiscussionTurn{
				TurnNumber: turn + 1,
				Responses:  []AIResponse{},
			}

			// OpenAI (ai1) responds
			fmt.Printf("%s %s's discussion turn %d:\n", color.Format(color.YELLOW, "Turn"), ai1.Name(openaiModel), turn+1)
			fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
			discussionResp, err := ai1.GenerateDiscussionResponse(ctx, prompt, discussionHistory, openaiModel, false)
			if err != nil {
				fmt.Printf("%s Error from %s: %v\n", color.Format(color.RED, "Error"), ai1.Name(openaiModel), err)
				cancel()
				break
			}
			// Clean the response to remove any AI name prefix
			cleanedResp := cleanResponse(discussionResp, ai1Name)
			fmt.Printf("%s %s\n", color.Format(color.GREEN, ai1.Name(openaiModel)), cleanedResp)
			fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
			discussionHistory += fmt.Sprintf("\n%s: %s", ai1Name, cleanedResp)
			discussionTurn.Responses = append(discussionTurn.Responses, AIResponse{
				AIName:   ai1Name,
				Response: cleanedResp,
			})

			// Grok (ai2) responds
			fmt.Printf("%s %s's discussion turn %d:\n", color.Format(color.YELLOW, "Turn"), ai2.Name(grokModel), turn+1)
			fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
			discussionResp, err = ai2.GenerateDiscussionResponse(ctx, prompt, discussionHistory, grokModel, false)
			if err != nil {
				fmt.Printf("%s Error from %s: %v\n", color.Format(color.RED, "Error"), ai2.Name(grokModel), err)
				cancel()
				break
			}
			// Clean the response to remove any AI name prefix
			cleanedResp = cleanResponse(discussionResp, ai2Name)
			fmt.Printf("%s %s\n", color.Format(color.GREEN, ai2.Name(grokModel)), cleanedResp)
			fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
			discussionHistory += fmt.Sprintf("\n%s: %s", ai2Name, cleanedResp)
			discussionTurn.Responses = append(discussionTurn.Responses, AIResponse{
				AIName:   ai2Name,
				Response: cleanedResp,
			})

			// Save discussion turn to session
			session.DiscussionTurns = append(session.DiscussionTurns, discussionTurn)

			// Prompt user to continue discussion or use commands
			for {
				fmt.Printf("(Enter to continue, or type 'h' for more..): ")
				userIntervention, err = reader.ReadString('\n')
				if err != nil {
					fmt.Printf("%s Error reading input: %v\n", color.Format(color.RED, "Error"), err)
					cancel()
					break
				}
				userIntervention = strings.TrimSpace(userIntervention)

				if userIntervention == "q" {
					fmt.Printf("%s Chat stopped.\n", color.Format(color.GREEN, "Exiting"))
					cancel()
					return // Exit the entire function
				}

				if userIntervention == "h" {
					displayHelp()
					continue // Stay in the inner loop to re-prompt
				}

				// Handle other commands or user input
				if userIntervention == "r" {
					ai1.ResetConversation()
					ai2.ResetConversation()
					codeBlock = ""
					codeLanguage = ""
					session.UserInputs = []string{}
					session.CodeBlocks = []CodeBlock{}
					session.AIConversations = []AIResponse{}
					session.DiscussionTurns = []DiscussionTurn{}
					fmt.Printf("%s Conversation history reset for both AIs.\n", color.Format(color.GREEN, "Reset"))
					cancel()
					break
				}

				if userIntervention == "c" {
					fmt.Printf("%s Enter programming language (e.g., 'go', 'python', or press Enter for 'none'):\n", color.Format(color.YELLOW, "Code Input"))
					language, err := reader.ReadString('\n')
					if err != nil {
						fmt.Printf("%s Error reading language input: %v\n", color.Format(color.RED, "Error"), err)
						cancel()
						continue
					}
					codeLanguage = strings.TrimSpace(language)

					fmt.Printf("%s Enter code block (end with '```' on a new line):\n", color.Format(color.YELLOW, "Code Input"))
					codeLines := []string{}
					for {
						line, err := reader.ReadString('\n')
						if err != nil {
							fmt.Printf("%s Error reading code input: %v\n", color.Format(color.RED, "Error"), err)
							break
						}
						line = strings.TrimRight(line, "\r\n")
						if line == "```" {
							break
						}
						codeLines = append(codeLines, line)
					}
					codeBlock = strings.Join(codeLines, "\n")
					if codeBlock == "" {
						fmt.Printf("%s No code entered. Continuing discussion.\n", color.Format(color.YELLOW, "Note"))
						codeLanguage = ""
						cancel()
						continue
					}

					if codeLanguage == "" || strings.EqualFold(codeLanguage, "none") {
						codeLanguage = "nolanguage"
					}

					// Save code block to session
					session.CodeBlocks = append(session.CodeBlocks, CodeBlock{
						Language: codeLanguage,
						Code:     codeBlock,
					})
					fmt.Printf("%s Code block saved (language: %s)\n", color.Format(color.GREEN, "Code Saved"), codeLanguage)

					// Update discussion history with new code block
					if codeBlock != "" {
						if codeLanguage != "" {
							discussionHistory = fmt.Sprintf("[Code Context]:\n```%s\n%s\n```\n\n%s", codeLanguage, codeBlock, discussionHistory)
						} else {
							discussionHistory = fmt.Sprintf("[Code Context]:\n```\n%s\n```\n\n%s", codeBlock, discussionHistory)
						}
					}
					cancel()
					continue
				}

				if userIntervention == "dc" {
					displayCodeBlocks(session)
					continue // Stay in the inner loop to re-prompt
				}

				if userIntervention != "" {
					fmt.Printf("%s User (you): %s\n", color.Format(color.BLUE, "User"), userIntervention)
					discussionHistory += fmt.Sprintf("\nUser: %s", userIntervention)
					// Save user intervention to session
					session.UserInputs = append(session.UserInputs, userIntervention)
				}

				cancel()
				break
			}
		}

		// Phase 3: Provide a short summary
		fmt.Printf("%s Discussion is over.\n", color.Format(color.YELLOW, "Result"))
	}
}

// AutoAIChat runs an automatic AI-to-AI discussion without user intervention
func AutoAIChat(ai1, ai2 AIClient, openaiModel, grokModel string, maxAITurns int) {
	fmt.Printf("%s Automatic AI-to-AI discussion with %s and %s started.\nNumber of discussion turns is %d. \nCommands: 'q' to quit, 'r' to reset history, 'h' for help, 'c' to enter code.\n",
		color.Format(color.GREEN, "Starting"), ai1.Name(openaiModel), ai2.Name(grokModel), maxAITurns)

	session := Session{
		Timestamp:       time.Now().UTC().Format(time.RFC3339),
		UserInputs:      []string{},
		CodeBlocks:      []CodeBlock{},
		AIConversations: []AIResponse{},
		DiscussionTurns: []DiscussionTurn{},
	}

	reader := bufio.NewReader(os.Stdin)
	var codeBlock string    // Store the current code block
	var codeLanguage string // Store the language for the code block

	defer func() {
		// Save session when the function exits
		if err := saveSessionToFile(session); err != nil {
			log.Printf("Error saving session: %v", err)
		}
	}()

	for {
		// Prompt user for initial input
		fmt.Printf("%s You (type intial prompt or command [check 'h']): ", color.Format(color.YELLOW, time.Now().UTC().Format(time.RFC1123)))
		userInput, err := reader.ReadString('\n')
		if err != nil {
			fmt.Printf("%s Error reading input: %v\n", color.Format(color.RED, "Error"), err)

			// break
			return // Exit the entire function on input error
		}
		userInput = strings.TrimSpace(userInput)

		if userInput == "q" {
			fmt.Printf("%s Chat stopped.\n", color.Format(color.GREEN, "Exiting"))
			break
		}

		if userInput == "h" {
			displayHelp()
			continue
		}

		if userInput == "r" {
			ai1.ResetConversation()
			ai2.ResetConversation()
			codeBlock = ""
			codeLanguage = ""
			// Reset session data (except timestamp)
			session.UserInputs = []string{}
			session.CodeBlocks = []CodeBlock{}
			session.AIConversations = []AIResponse{}
			session.DiscussionTurns = []DiscussionTurn{}
			fmt.Printf("%s Conversation history reset for both AIs.\n", color.Format(color.GREEN, "Reset"))
			continue
		}
		if userInput == "c" {
			fmt.Printf("%s Enter programming language (e.g., 'go', 'python', or press Enter for 'none'):\n", color.Format(color.YELLOW, "Code Input"))
			language, err := reader.ReadString('\n')
			if err != nil {
				fmt.Printf("%s Error reading language input: %v\n", color.Format(color.RED, "Error"), err)
				continue
			}
			codeLanguage = strings.TrimSpace(language)

			fmt.Printf("%s Enter code block (end with '```' on a new line):\n", color.Format(color.YELLOW, "Code Input"))
			codeLines := []string{}
			for {
				line, err := reader.ReadString('\n')
				if err != nil {
					fmt.Printf("%s Error reading code input: %v\n", color.Format(color.RED, "Error"), err)
					break
				}
				line = strings.TrimRight(line, "\r\n")
				if line == "```" {
					break
				}
				codeLines = append(codeLines, line)
			}
			codeBlock = strings.Join(codeLines, "\n")
			if codeBlock == "" {
				fmt.Printf("%s No code entered. Returning to main prompt.\n", color.Format(color.YELLOW, "Note"))
				codeLanguage = ""
				continue
			}

			if codeLanguage == "" || strings.EqualFold(codeLanguage, "none") {
				codeLanguage = "nolanguage"
			}

			// Save code block to session
			session.CodeBlocks = append(session.CodeBlocks, CodeBlock{
				Language: codeLanguage,
				Code:     codeBlock,
			})
			fmt.Printf("%s Code block saved (language: %s)\n", color.Format(color.GREEN, "Code Saved"),
				codeLanguage)
			continue
			// fmt.Printf("%s Code block saved (language: %s):\n%s\n", color.Format(color.GREEN, "Code Saved"),
			// 	codeLanguage, codeBlock)
			// continue
		}

		if userInput == "dc" {
			displayCodeBlocks(session)
			continue
		}

		if userInput == "" {
			continue
		}

		// Save user input to session
		session.UserInputs = append(session.UserInputs, userInput)

		// Combine user input with code block if available
		prompt := userInput
		if codeBlock != "" {
			if codeLanguage != "" {
				prompt = fmt.Sprintf("%s\n\n[Code Context]:\n```%s\n%s\n```", userInput, codeLanguage, codeBlock)
			} else {
				prompt = fmt.Sprintf("%s\n\n[Code Context]:\n```\n%s\n```", userInput, codeBlock)
			}
		}

		// Phase 1: Both AIs respond to the user prompt concurrently
		ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
		var wg sync.WaitGroup
		type aiResponse struct {
			name     string
			response string
			err      error
		}
		responses := make(chan aiResponse, 2)

		// Launch goroutine for AI1 (OpenAI)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fmt.Printf("%s Waiting for %s...\n", color.Format(color.YELLOW, "Processing"), ai1.Name(openaiModel))
			resp, err := ai1.GenerateResponse(ctx, prompt, openaiModel)
			responses <- aiResponse{name: ai1.Name(openaiModel), response: resp, err: err}
		}()

		// Launch goroutine for AI2 (Grok)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fmt.Printf("%s Waiting for %s...\n", color.Format(color.YELLOW, "Processing"), ai2.Name(grokModel))
			resp, err := ai2.GenerateResponse(ctx, prompt, grokModel)
			responses <- aiResponse{name: ai2.Name(grokModel), response: resp, err: err}
		}()

		// Wait for both responses and close channel
		go func() {
			wg.Wait()
			close(responses)
		}()

		// Collect initial responses
		ai1Response, ai2Response := "", ""
		ai1Name, ai2Name := ai1.Name(openaiModel), ai2.Name(grokModel)
		for resp := range responses {
			if resp.err != nil {
				fmt.Printf("%s Error from %s: %v\n", color.Format(color.RED, "Error"), resp.name, resp.err)
				continue
			}
			// Clean the response to remove any AI name prefix
			cleanedResp := cleanResponse(resp.response, resp.name)
			fmt.Printf("%s %s\n", color.Format(color.GREEN, resp.name), cleanedResp)
			fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
			if resp.name == ai1Name {
				ai1Response = cleanedResp
			} else if resp.name == ai2Name {
				ai2Response = cleanedResp
			}
		}
		// Cancel context after both responses are collected
		cancel()

		// Save initial responses to session.AIConversations
		if ai1Response != "" {
			session.AIConversations = append(session.AIConversations, AIResponse{
				AIName:   ai1Name,
				Response: ai1Response,
			})
		}
		if ai2Response != "" {
			session.AIConversations = append(session.AIConversations, AIResponse{
				AIName:   ai2Name,
				Response: ai2Response,
			})
		}

		// Skip discussion if either response failed
		if ai1Response == "" || ai2Response == "" {
			fmt.Printf("%s Discussion skipped due to missing responses.\n", color.Format(color.YELLOW, "Note"))
			continue
		}

		// Phase 2: Automatic AI-to-AI discussion
		fmt.Printf("%s Starting automatic AI-to-AI discussion...\n", color.Format(color.YELLOW, "Discussion"))
		fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
		discussionHistory := fmt.Sprintf("%s: %s\n%s: %s", ai1Name, ai1Response, ai2Name, ai2Response)
		if codeBlock != "" {
			if codeLanguage != "" {
				discussionHistory = fmt.Sprintf("[Code Context]:\n```%s\n%s\n```\n\n%s", codeLanguage, codeBlock, discussionHistory)
			} else {
				discussionHistory = fmt.Sprintf("[Code Context]:\n```\n%s\n```\n\n%s", codeBlock, discussionHistory)
			}
		}

		for turn := 0; turn < maxAITurns; turn++ {
			// '30*time.Second' causes 'context deadline exceeded'
			ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
			discussionTurn := DiscussionTurn{
				TurnNumber: turn + 1,
				Responses:  []AIResponse{},
			}

			// OpenAI (ai1) responds
			fmt.Printf("%s %s's discussion turn %d:\n", color.Format(color.YELLOW, "Turn"), ai1.Name(openaiModel), turn+1)
			fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
			discussionResp, err := ai1.GenerateDiscussionResponse(ctx, prompt, discussionHistory, openaiModel, true)
			if err != nil {
				fmt.Printf("%s Error from %s: %v\n", color.Format(color.RED, "Error"), ai1.Name(openaiModel), err)
				cancel()
				break
			}
			// Clean the response to remove any AI name prefix
			cleanedResp := cleanResponse(discussionResp, ai1Name)
			fmt.Printf("%s %s\n", color.Format(color.GREEN, ai1.Name(openaiModel)), cleanedResp)
			fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
			discussionHistory += fmt.Sprintf("\n%s: %s", ai1Name, cleanedResp)
			discussionTurn.Responses = append(discussionTurn.Responses, AIResponse{
				AIName:   ai1Name,
				Response: cleanedResp,
			})

			// Grok (ai2) responds
			fmt.Printf("%s %s's discussion turn %d:\n", color.Format(color.YELLOW, "Turn"), ai2.Name(grokModel), turn+1)
			fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
			discussionResp, err = ai2.GenerateDiscussionResponse(ctx, prompt, discussionHistory, grokModel, true)
			if err != nil {
				fmt.Printf("%s Error from %s: %v\n", color.Format(color.RED, "Error"), ai2.Name(grokModel), err)
				cancel()
				break
			}
			// Clean the response to remove any AI name prefix
			cleanedResp = cleanResponse(discussionResp, ai2Name)
			fmt.Printf("%s %s\n", color.Format(color.GREEN, ai2.Name(grokModel)), cleanedResp)
			fmt.Printf("%s\n", color.Format(color.PURPLE, "------------------------------------------------"))
			discussionHistory += fmt.Sprintf("\n%s: %s", ai2Name, cleanedResp)
			discussionTurn.Responses = append(discussionTurn.Responses, AIResponse{
				AIName:   ai2Name,
				Response: cleanedResp,
			})

			// Save discussion turn to session
			session.DiscussionTurns = append(session.DiscussionTurns, discussionTurn)
			cancel()
		}

		// Phase 3: Provide a short summary
		fmt.Printf("%s Discussion is over.\n", color.Format(color.YELLOW, "Result"))
	}
}

// displayCodeBlocks prints all code blocks in a Session.
func displayCodeBlocks(session Session) {
	if len(session.CodeBlocks) == 0 {
		fmt.Printf("%s No code blocks saved in this session.\n", color.Format(color.YELLOW, "Note"))
		return
	}
	fmt.Printf("%s Saved code blocks:\n", color.Format(color.GREEN, "Code Blocks"))
	for i, cb := range session.CodeBlocks {
		fmt.Printf("%s [%d] language: %s\n", color.Format(color.BLUE, "Code"), i+1, cb.Language)
		fmt.Printf("-----\n%s\n-----\n", cb.Code)
	}
}

func main() {
	openaiKey := os.Getenv("API_KEY")
	if openaiKey == "" {
		log.Fatal("Please set the API_KEY environment variable for OpenAI")
	}
	xaiKey := os.Getenv("XAI_API_KEY")
	if xaiKey == "" {
		log.Fatal("Please set the XAI_API_KEY environment variable for xAI")
	}

	openaiModel := os.Getenv("OPENAI_MODEL")
	if openaiModel == "" {
		openaiModel = defaultModel
	}
	grokModel := os.Getenv("GROK_MODEL")
	if grokModel == "" {
		grokModel = defaultGrokModel
	}
	grokEndpoint := os.Getenv("GROK_ENDPOINT")
	if grokEndpoint == "" {
		grokEndpoint = defaultEndpoint
	}

	// Maximum turns for AI-to-AI discussion
	maxAITurnsStr := os.Getenv("AI_TURNS")
	if maxAITurnsStr == "" {
		maxAITurnsStr = defaultAITurns
	}
	maxAITurns, err := strconv.Atoi(maxAITurnsStr)
	if err != nil {
		log.Fatalf("Invalid AI_TURNS value: %v", err)
	}

	autoMode := flag.Bool("auto", false, "Run in automatic AI-to-AI discussion mode")
	// todo
	// reqDebug := flag.Bool("debug", false, "Debug mode - show HTTP requests details")
	flag.Parse()

	openaiClient := NewOpenAIClientAdapter(openaiKey)
	grokClient := NewGrokClient(xaiKey, grokEndpoint)

	ShowBanner()

	if *autoMode {
		AutoAIChat(openaiClient, grokClient, openaiModel, grokModel, maxAITurns)
	} else {
		ParallelUserAIChatWithAgreement(openaiClient, grokClient, openaiModel, grokModel, maxAITurns)
	}
}
