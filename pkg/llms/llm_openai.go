package llms

import (
	"context"
	"log"
	"time"

	"github.com/tmc/langchaingo/schema"

	"github.com/tmc/langchaingo/llms"

	"github.com/getzep/zep/config"
	"github.com/getzep/zep/pkg/models"
	"github.com/pkoukk/tiktoken-go"
	"github.com/tmc/langchaingo/llms/openai"
)

const OpenAIAPITimeout = 90 * time.Second
const OpenAIAPIKeyNotSetError = "ZEP_OPENAI_API_KEY is not set" //nolint:gosec
const MaxOpenAIAPIRequestAttempts = 5

var _ models.ZepLLM = &ZepOpenAILLM{}

func NewOpenAILLM(ctx context.Context, llmConfig *config.LLM) (*ZepOpenAILLM, error) {
	zllm := &ZepOpenAILLM{}
	err := zllm.Init(ctx, llmConfig)
	if err != nil {
		return nil, err
	}
	return zllm, nil
}

type ZepOpenAILLM struct {
	llm *openai.Chat
	tkm *tiktoken.Tiktoken
}

func (zllm *ZepOpenAILLM) Init(_ context.Context, llmConfig *config.LLM) error {
	// Initialize the Tiktoken client
	encoding := "cl100k_base"
	tkm, err := tiktoken.GetEncoding(encoding)
	if err != nil {
		return err
	}
	zllm.tkm = tkm

	options, err := zllm.configureClient(llmConfig)
	if err != nil {
		return err
	}

	// Create a new client instance with options
	llm, err := openai.NewChat(options...)
	if err != nil {
		return err
	}
	zllm.llm = llm

	return nil
}

func (zllm *ZepOpenAILLM) Call(ctx context.Context,
	prompt string,
	options ...llms.CallOption,
) (string, error) {
	// If the LLM is not initialized, return an error
	if zllm.llm == nil {
		return "", NewLLMError(InvalidLLMModelError, nil)
	}

	if len(options) == 0 {
		options = append(options, llms.WithTemperature(DefaultTemperature))
	}

	thisCtx, cancel := context.WithTimeout(ctx, OpenAIAPITimeout)
	defer cancel()

	messages := []schema.ChatMessage{schema.SystemChatMessage{Content: prompt}}

	completion, err := zllm.llm.Call(thisCtx, messages, options...)
	if err != nil {
		return "", err
	}

	return completion.GetContent(), nil
}

func (zllm *ZepOpenAILLM) EmbedTexts(ctx context.Context, texts []string) ([][]float32, error) {
	// If the LLM is not initialized, return an error
	if zllm.llm == nil {
		return nil, NewLLMError(InvalidLLMModelError, nil)
	}

	thisCtx, cancel := context.WithTimeout(ctx, OpenAIAPITimeout)
	defer cancel()

	embeddings, err := zllm.llm.CreateEmbedding(thisCtx, texts)
	if err != nil {
		return nil, NewLLMError("error while creating embedding", err)
	}

	return embeddings, nil
}

// GetTokenCount returns the number of tokens in the text
func (zllm *ZepOpenAILLM) GetTokenCount(text string) (int, error) {
	return len(zllm.tkm.Encode(text, nil, nil)), nil
}

func (zllm *ZepOpenAILLM) configureClient(llmConfig *config.LLM) ([]openai.Option, error) {
	// Retrieve the OpenAIAPIKey from configuration
	apiKey := llmConfig.OpenAIAPIKey
	// If the key is not set, log a fatal error and exit
	if apiKey == "" {
		log.Fatal(OpenAIAPIKeyNotSetError)
	}
	if llmConfig.AzureOpenAIEndpoint != "" && llmConfig.OpenAIEndpoint != "" {
		log.Fatal("only one of AzureOpenAIEndpoint or OpenAIEndpoint can be set")
	}

	retryableHTTPClient := NewRetryableHTTPClient(MaxOpenAIAPIRequestAttempts, OpenAIAPITimeout)

	options := make([]openai.Option, 0)
	options = append(
		options,
		openai.WithHTTPClient(retryableHTTPClient.StandardClient()),
		openai.WithModel(llmConfig.Model),
		openai.WithToken(apiKey),
	)

	switch {
	case llmConfig.AzureOpenAIEndpoint != "":
		// Check configuration for AzureOpenAIEndpoint; if it's set, use the DefaultAzureConfig
		// and provided endpoint Path
		options = append(
			options,
			openai.WithAPIType(openai.APITypeAzure),
			openai.WithBaseURL(llmConfig.AzureOpenAIEndpoint),
		)
		if llmConfig.AzureOpenAIModel.EmbeddingDeployment != "" {
			options = append(
				options,
				openai.WithEmbeddingModel(llmConfig.AzureOpenAIModel.EmbeddingDeployment),
			)
		}
	case llmConfig.OpenAIEndpoint != "":
		// If an alternate OpenAI-compatible endpoint Path is set, use this as the base Path for requests
		options = append(
			options,
			openai.WithBaseURL(llmConfig.OpenAIEndpoint),
		)
	case llmConfig.OpenAIOrgID != "":
		options = append(options, openai.WithOrganization(llmConfig.OpenAIOrgID))
	}

	return options, nil
}
