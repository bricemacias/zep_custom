package llms

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/getzep/zep/pkg/models"

	"github.com/hashicorp/go-retryablehttp"

	"github.com/getzep/zep/config"

	"github.com/getzep/zep/internal"
)

const DefaultTemperature = 0.0
const InvalidLLMModelError = "llm model is not set or is invalid"

var log = internal.GetLogger()

func handleOpenAIClient(ctx context.Context, llmConfig *config.LLM, clientType string) (models.ZepLLM, error) {
	// Azure OpenAI model names can't be validated by any hard-coded models
	// list as it is configured by custom deployment name that may or may not match the model name.
	// We will copy the Model name value down to AzureOpenAI LLM Deployment
	// to assume user deployed base model with matching deployment name as
	// advised by Microsoft, but still support custom models or otherwise-named
	// base model.
	if llmConfig.AzureOpenAIEndpoint != "" {
		if llmConfig.AzureOpenAIModel.LLMDeployment != "" {
			llmConfig.Model = llmConfig.AzureOpenAIModel.LLMDeployment
		}
		// if custom OpenAI Endpoint is set, do not validate model name
		if cfg.LLM.OpenAIEndpoint != "" {
			return NewOpenAILLM(ctx, cfg)
		}
		// Otherwise, validate model name
		if _, ok := ValidOpenAILLMs[cfg.LLM.Model]; !ok {
			return nil, fmt.Errorf(
				"invalid llm deployment for %s, deployment name is required",
				llmConfig.Service,
			)
		}

		// EmbeddingsDeployment is only required if Zep is also configured to use
		// OpenAI embeddings for document or message extractors
		if llmConfig.AzureOpenAIModel.EmbeddingDeployment == "" && useOpenAIEmbeddings(cfg) {
			return nil, fmt.Errorf(
				"invalid embeddings deployment for %s, deployment name is required",
				llmConfig.Service,
			)
		}
		return NewOpenAILLM(ctx, llmConfig)
	}

	isUsingCustomLLMEndpoint = cfg.LLM.OpenAIEndpoint != "" && cfg.OpenAIEmbeddings.Enabled
	_, isValidOpenAILLM := ValidOpenAILLMs[llmConfig.Model]
	_, isValidOpenSourceLLM := isUsingCustomLLMEndpoint && ValidOpenSourceLLMs[llmConfig.Model]
	isValidLLM = isValidOpenAILLM || isValidOpenSourceLLM
	err := fmt.Errorf(
		"invalid llm model \"%s\" for %s",
		llmConfig.Model,
		llmConfig.Service,
	)

  // Even when only using the OpenAI client for embeddings, 
	// the LLM model must be set to a valid OpenAI model.
	if clientType = "embeddings" && !isValidOpenAILLM {
		return nil, err
	} else if !isValidLLM{
		return nil, err
	}

	return NewOpenAILLM(ctx, llmConfig)
}


func NewLLMClient(ctx context.Context, cfg *config.Config) (models.ZepLLM, error) {
	llmConfig := cfg.LLM

	switch llmConfig.Service {
	case "openai":
		return handleOpenAIClient(ctx, llmConfig, "llm")
	case "anthropic":
		if _, ok := ValidAnthropicLLMs[llmConfig.Model]; !ok {
			return nil, fmt.Errorf(
				"invalid llm model \"%s\" for %s",
				llmConfig.Model,
				llmConfig.Service,
			)
		}
		return NewAnthropicLLM(ctx, cfg)
	case "":
		// for backward compatibility
		return NewOpenAILLM(ctx, llmConfig)
	default:
		return nil, fmt.Errorf("invalid LLM service: %s", llmConfig.Service)
	}
}


func NewEmbeddingsClient(ctx context.Context, cfg *config.Config) (models.ZepLLM, error) {
	llmConfig := cfg.OpenAIEmbeddings.Client

	switch llmConfig.Service {
	case "openai":
		return handleOpenAIClient(ctx, llmConfig, "embeddings")
	case "":
		// for backward compatibility
		return NewOpenAILLM(ctx, llmConfig)
	default:
		return nil, fmt.Errorf("invalid OpenAI Embeddings service: %s", llmConfig.Service)
	}
}

type LLMError struct {
	message       string
	originalError error
}

func (e *LLMError) Error() string {
	return fmt.Sprintf("llm error: %s (original error: %v)", e.message, e.originalError)
}

func NewLLMError(message string, originalError error) *LLMError {
	return &LLMError{message: message, originalError: originalError}
}

var ValidOpenAILLMs = map[string]bool{
	"gpt-3.5-turbo":     true,
	"gpt-4":             true,
	"gpt-3.5-turbo-16k": true,
	"gpt-4-32k":         true,
}

var ValidOpenSourceLLMs = map[string]bool{
	"meta-llama/Llama-2-7b-chat-hf":  true,
	"meta-llama/Llama-2-13b-chat-hf": true,
	"meta-llama/Llama-2-70b-chat-hf": true,
}

var ValidAnthropicLLMs = map[string]bool{
	"claude-instant-1": true,
	"claude-2":         true,
}

var ValidLLMMap = internal.MergeMaps(ValidOpenAILLMs, ValidAnthropicLLMs)

var MaxLLMTokensMap = map[string]int{
	"gpt-3.5-turbo":                  4096,
	"gpt-3.5-turbo-16k":              16_384,
	"gpt-4":                          8192,
	"gpt-4-32k":                      32_768,
	"claude-instant-1":               100_000,
	"claude-2":                       100_000,
	"meta-llama/Llama-2-7b-chat-hf":  4096
	"meta-llama/Llama-2-13b-chat-hf": 4096
	"meta-llama/Llama-2-70b-chat-hf": 4096
}

func GetLLMModelName(cfg *config.Config) (string, error) {
	llmModel := cfg.LLM.Model
	// Don't validate if custom OpenAI endpoint or Azure OpenAI endpoint is set
	if cfg.LLM.OpenAIEndpoint != "" || cfg.LLM.AzureOpenAIEndpoint != "" {
		return llmModel, nil
	}
	if llmModel == "" || !ValidLLMMap[llmModel] {
		return "", NewLLMError(InvalidLLMModelError, nil)
	}
	return llmModel, nil
}

func Float64ToFloat32Matrix(in [][]float64) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = make([]float32, len(in[i]))
		for j, v := range in[i] {
			out[i][j] = float32(v)
		}
	}

	return out
}

func NewRetryableHTTPClient(retryMax int, timeout time.Duration) *retryablehttp.Client {
	retryableHTTPClient := retryablehttp.NewClient()
	retryableHTTPClient.RetryMax = retryMax
	retryableHTTPClient.HTTPClient.Timeout = timeout
	retryableHTTPClient.Logger = log
	retryableHTTPClient.Backoff = retryablehttp.DefaultBackoff
	retryableHTTPClient.CheckRetry = retryPolicy

	return retryableHTTPClient
}

// retryPolicy is a retryablehttp.CheckRetry function. It is used to determine
// whether a request should be retried or not.
func retryPolicy(ctx context.Context, resp *http.Response, err error) (bool, error) {
	// do not retry on context.Canceled or context.DeadlineExceeded
	if ctx.Err() != nil {
		return false, ctx.Err()
	}

	// Do not retry 400 errors as they're used by OpenAI to indicate maximum
	// context length exceeded
	if resp != nil && resp.StatusCode == 400 {
		return false, err
	}

	shouldRetry, _ := retryablehttp.DefaultRetryPolicy(ctx, resp, err)
	return shouldRetry, nil
}

// useOpenAIEmbeddings is true if OpenAI embeddings are enabled
func useOpenAIEmbeddings(cfg *config.Config) bool {
	switch {
	case cfg.Extractors.Messages.Embeddings.Enabled:
		return cfg.Extractors.Messages.Embeddings.Service == "openai"
	case cfg.Extractors.Documents.Embeddings.Enabled:
		return cfg.Extractors.Documents.Embeddings.Service == "openai"
	}

	return false
}
