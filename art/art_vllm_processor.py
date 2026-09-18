import base64
import io
from typing import List

import numpy as np
import soundfile as sf
from openai import OpenAI
from openai._exceptions import APIError, RateLimitError, APITimeoutError


class vLLMProcessor:
    """
    Wrapper around an OpenAI-compatible vLLM server.

    Args:
        model_name (str): Name of the model served by vLLM.
        api_key (str): API key (often 'EMPTY' for local vLLM).
        base_url (str): Base URL of the OpenAI-compatible API (e.g., 'http://localhost:8000/v1').
        max_completion_tokens (int): Upper bound on generated tokens.
        descriptive (bool): If False, prompts the model to reply strictly with 'Yes' or 'No'. If
            True, omits that constraint (free-form answer).
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        max_completion_tokens: int,
        descriptive: bool,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.descriptive = descriptive
        self._buf = io.BytesIO()

    def _encode_audio_b64(self, waveform: np.ndarray, sampling_rate: int) -> str:
        """
        Encode a waveform as WAV and return base64.

        Args:
            waveform (np.ndarray): Numpy array (shape [T] or [T, C]) of audio samples.
            sampling_rate (int): Sampling rate in Hz.
        
        Returns:
            str: Base64-encoded WAV bytes.
        """
        b = self._buf
        b.seek(0)
        b.truncate(0)
        sf.write(b, waveform, sampling_rate, format="WAV")
        view = b.getbuffer()
        try:
            b64 = base64.b64encode(view).decode("ascii")
        finally:
            view.release()
        return b64
    
    def _get_messages(self, audio_b64: str) -> List[dict]:
        """Build an OpenAI-chat message payload with an input_audio."""
        text_prompt = "Answer the question from the audio."
        if not self.descriptive:
            text_prompt += " Answer only \"Yes\" or \"No\"."
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    },
                ],
            },
        ]
    
    def infer_sample(self, x: dict) -> str:
        audio = x["audio"]
        audio_b64 = self._encode_audio_b64(audio["array"], audio["sampling_rate"])
        messages = self._get_messages(audio_b64)

        try:
            out = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                max_completion_tokens=self.max_completion_tokens,
            )
        except (APITimeoutError, RateLimitError, APIError) as e:
            raise RuntimeError(f"Inference failed: {e}") from e
        
        choices = getattr(out, "choices", None)
        if not isinstance(choices, list) or not choices:
            self._invalid_response(out, "missing or empty choices")
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            self._invalid_response(out, "missing or empty message content")
        return content

    @staticmethod
    def _invalid_response(response, reason: str) -> None:
        """Expose server diagnostics instead of failing while indexing a malformed response."""
        try:
            details = response.model_dump_json(exclude_none=True)
        except (AttributeError, TypeError, ValueError):
            details = repr(response)
        if len(details) > 2000:
            details = details[:2000] + "... [truncated]"
        raise RuntimeError(
            f"Invalid chat-completion response ({reason}). Server response: {details}. "
            "Check the vLLM server log and that --base-url points to its /v1 API."
        )
