# Environment Variables

## Required Variables

### `OPENAI_API_KEY`
- **Purpose**: Authentication for OpenAI API calls
- **Used in**:
  - `core/agent.py`: Line 8, no default
  - `core/classifier.py`: Line 1, no default
  - `core/resource_detector.py`: Line 33, no default
- **Validation**: Raises ValueError if not set

### `DISCORD_TOKEN`
- **Purpose**: Authentication for Discord bot
- **Used in**: `core/bot.py`: Line 24, no default
- **Validation**: Raises ValueError if not set

## Optional Variables

### `GPT_MODEL`
- **Purpose**: Specifies which OpenAI model to use
- **Default**: "gpt-4-turbo"
- **Used in**:
  - `core/agent.py`: Line 10
  - `core/classifier.py`: Line 2
  - `core/resource_detector.py`: Line 34

## Environment Loading

The application uses `python-dotenv` to load environment variables from `.env` files:
- `core/agent.py`: Line 7
- `core/bot.py`: Line 23

## Security Notes

1. API keys are loaded at module level but should be loaded lazily
2. No environment variable validation beyond presence checks
3. No type conversion or sanitization of environment values
4. No secrets rotation mechanism visible in code

## Recommendations

1. Add validation for API key formats
2. Implement secrets rotation
3. Add type conversion for numeric environment variables
4. Consider using a secrets management service
5. Add environment variable documentation to README.md 