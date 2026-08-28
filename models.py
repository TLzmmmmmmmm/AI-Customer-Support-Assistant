from typing import Literal

from pydantic import BaseModel, Field, model_validator

from config import (
    MAX_CONVERSATION_CHARACTERS,
    MAX_CONVERSATION_MESSAGES,
    MAX_MESSAGE_CHARACTERS,
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]

    content: str = Field(
        min_length=1,
        max_length=MAX_MESSAGE_CHARACTERS,
    )


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        min_length=1,
        max_length=MAX_CONVERSATION_MESSAGES,
    )

    @model_validator(mode="after")
    def validate_total_content_length(self):
        total_characters = sum(
            len(message.content)
            for message in self.messages
        )

        if total_characters > MAX_CONVERSATION_CHARACTERS:
            raise ValueError(
                "Conversation is too long"
            )

        return self