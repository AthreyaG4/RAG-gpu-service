from pydantic import BaseModel, Field


class SummarizationRequest(BaseModel):
    chunk_text: str = Field(..., description="The text chunk to be summarized.")
    image_urls: list[str] = Field(
        ..., description="This are the urls of the images associated with this chunk"
    )


class SummarizationResponse(BaseModel):
    summary_text: str = Field(..., description="The generated summary text.")


class EmbeddingRequest(BaseModel):
    summarized_text: str = Field(..., description="The text to be embedded.")


class EmbeddingResponse(BaseModel):
    embedding_vector: list[float] = Field(
        ..., description="The generated embedding vector."
    )
