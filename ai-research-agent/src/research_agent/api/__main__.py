"""``python -m research_agent.api`` starts the API server."""
import uvicorn

from research_agent.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("research_agent.api.main:create_app", factory=True, host=s.api_host, port=s.api_port)
