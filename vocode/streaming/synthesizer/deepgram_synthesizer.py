import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Optional, Union, Tuple
import io
import aiohttp
from vocode import getenv
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    tracer,
)
from vocode.streaming.models.synthesizer import (
    DeepgramSynthesizerConfig,
    SynthesizerType,
)
from vocode.streaming.agent.bot_sentiment_analyser import BotSentiment
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.utils.mp3_helper import decode_mp3

DEEPGRAM_BASE_URL = "https://api.deepgram.com/v1/speak"

class DeepgramSynthesizer(BaseSynthesizer[DeepgramSynthesizerConfig]):
    def __init__(
        self,
        synthesizer_config: DeepgramSynthesizerConfig,
        logger: Optional[logging.Logger] = None,
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
    ):
        super().__init__(synthesizer_config, aiohttp_session)

        self.api_key = synthesizer_config.api_key or getenv("DEEPGRAM_API_KEY")
        self.model = synthesizer_config.model or "aura-asteria-en"

    async def create_speech(
        self,
        message: BaseMessage,
        chunk_size: int,
        bot_sentiment: Optional[BotSentiment] = None,
    ) -> SynthesisResult:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}",
        }
        data = {"text": message.text}
        params = {"model": self.model}

        create_speech_span = tracer.start_span(
            f"synthesizer.{SynthesizerType.DEEPGRAM.value.split('_', 1)[-1]}.create_total",
        )

        async with self.aiohttp_session.post(
            DEEPGRAM_BASE_URL,
            headers=headers,
            json=data,
            params=params,
            timeout=aiohttp.ClientTimeout(total=15),
        ) as response:
            if not response.ok:
                raise Exception(f"Deepgram API returned {response.status} status code")

            async def stream_response():
                async for chunk in response.content.iter_any():
                    yield chunk

            create_speech_span.end()
            convert_span = tracer.start_span(
                f"synthesizer.{SynthesizerType.DEEPGRAM.value.split('_', 1)[-1]}.convert",
            )

            output_bytes_io = io.BytesIO()
            async for chunk in stream_response():
                output_bytes_io.write(chunk)
                output_bytes_io.seek(0)
                audio_data = output_bytes_io.read()
                output_bytes_io.seek(0)
                output_bytes_io.truncate(0)

                if len(audio_data) >= chunk_size:
                    yield audio_data[:chunk_size]
                    audio_data = audio_data[chunk_size:]

            if audio_data:
                yield audio_data

            output_bytes_io.close()
            convert_span.end()
