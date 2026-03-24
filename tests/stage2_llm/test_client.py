import pytest
from unittest.mock import patch, AsyncMock
from llm.client import OllamaClient, LLMError

@pytest.fixture
def client():
    return OllamaClient(host="http://localhost:11434", model="qwen2.5:0.5b", timeout=5.0)

@pytest.mark.asyncio
async def test_generate_returns_string(client):
    mock_response = {"response": "Alice goes to the library."}
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = lambda: None
        result = await client.generate("What does Alice do?")
    assert isinstance(result, str)
    assert len(result) > 0

@pytest.mark.asyncio
async def test_generate_uses_fallback_on_timeout(client):
    with patch("httpx.AsyncClient.post", side_effect=Exception("timeout")):
        result = await client.generate("test prompt", fallback="reading quietly")
    assert result == "reading quietly"

@pytest.mark.asyncio
async def test_generate_raises_without_fallback(client):
    with patch("httpx.AsyncClient.post", side_effect=Exception("timeout")):
        with pytest.raises(LLMError):
            await client.generate("test prompt")

@pytest.mark.asyncio
async def test_score_importance_returns_float(client):
    mock_response = {"response": "7"}
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = lambda: None
        score = await client.score_importance("Alice got accepted to university")
    assert 1.0 <= score <= 10.0

@pytest.mark.asyncio
async def test_score_importance_fallback_on_parse_error(client):
    mock_response = {"response": "very important"}  # not a number
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status = lambda: None
        score = await client.score_importance("test event")
    assert score == 5.0  # default fallback

@pytest.mark.asyncio
async def test_embed_returns_list(client):
    result = await client.embed("hello world")
    assert isinstance(result, list)
    assert len(result) == 384  # all-MiniLM-L6-v2 dim
