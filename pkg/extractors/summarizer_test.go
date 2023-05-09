package extractors

import (
	"context"
	"github.com/danielchalef/zep/pkg/llms"
	"github.com/danielchalef/zep/pkg/memorystore"
	"github.com/danielchalef/zep/pkg/models"
	"github.com/danielchalef/zep/test"
	"github.com/google/uuid"
	"github.com/jinzhu/copier"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestSummarize(t *testing.T) {
	ctx := context.Background()

	db := memorystore.NewPostgresConn(test.TestDsn)
	defer db.Close()
	memorystore.CleanDB(t, db)

	cfg, err := test.NewTestConfig()
	assert.NoError(t, err)

	appState := &models.AppState{Config: cfg}

	store, err := memorystore.NewPostgresMemoryStore(appState, db)
	assert.NoError(t, err)

	appState.OpenAIClient = llms.CreateOpenAIClient(cfg)
	appState.MemoryStore = store

	windowSize := 10
	newMessageCountAfterSummary := windowSize / 2

	messages := make([]models.Message, len(test.TestMessages))
	err = copier.Copy(&messages, &test.TestMessages)
	assert.NoError(t, err)

	messages = messages[:windowSize+2]
	for i := range messages {
		messages[i].UUID = uuid.New()
	}

	newestMessageToSummarizeIndex := len(
		messages,
	) - newMessageCountAfterSummary - 1 // the seventh-oldest message, leaving 5 messages after it
	newSummaryPointUUID := messages[newestMessageToSummarizeIndex].UUID

	tests := []struct {
		name     string
		messages []models.Message
		summary  *models.Summary
	}{
		{
			name:     "With an existing summary",
			messages: messages,
			summary: &models.Summary{
				Content:    "Existing summary content",
				TokenCount: 10,
			},
		},
		{
			name:     "With a nil-value passed as the summary argument",
			messages: messages,
			summary:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			newSummary, err := summarize(ctx, appState, windowSize, tt.messages, tt.summary, 0)
			assert.NoError(t, err)

			assert.Equal(t, newSummaryPointUUID, newSummary.SummaryPointUUID)
			assert.NotEmpty(t, newSummary.Content)
			assert.True(t, newSummary.TokenCount > 0)
		})
	}
}