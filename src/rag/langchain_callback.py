from langchain_core.callbacks.base import BaseCallbackHandler

class UsageTrackingCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.completion_token_details = None
        self.total_cost = 0.0
        self.calls = 0
        self.completion_token_details = None

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output
        if usage:
            self.prompt_tokens += usage["token_usage"]["prompt_tokens"]
            self.completion_tokens += usage["token_usage"]["completion_tokens"]
            self.completion_token_details = usage["token_usage"]["completion_tokens_details"]
            self.total_tokens += usage["token_usage"]["total_tokens"]
            self.calls += 1
