"""ASGI entry point: `uvicorn neuralscope_api.server:app`. Settings come from the environment."""

from neuralscope_api.main import create_app

app = create_app()
