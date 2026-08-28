from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

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

    @field_validator("content", mode="before")
    @classmethod
    def strip_content(cls, value):
        if isinstance(value, str):
            return value.strip()

        return value


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

    @model_validator(mode="after")
    def validate_role_sequence(self):
        if self.messages[0].role != "user":
            raise ValueError(
                "Conversation must start with a user message"
            )

        for previous, current in zip(
            self.messages,
            self.messages[1:],
        ):
            if previous.role == current.role:
                raise ValueError(
                    "Conversation roles must alternate"
                )

        if self.messages[-1].role != "user":
            raise ValueError(
                "Conversation must end with a user message"
            )

        return self
