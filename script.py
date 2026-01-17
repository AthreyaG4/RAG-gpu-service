import asyncio
import torch
import time
import logging
from typing import Optional
from collections import deque
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from schemas import (
    SummarizationRequest,
    EmbeddingRequest,
    SummarizationResponse,
    EmbeddingResponse,
)

summarizing_model_id = "/repository"
# summarizing_model_id = "Qwen/Qwen3-VL-2B-Instruct"
embedding_model_id = "all-MiniLM-L6-v2"

BATCH_SIZE = 8
BATCH_TIMEOUT = 0.05

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when attempting to use the model before it is loaded."""


class ModelManager:
    def __init__(
        self,
        summarizing_model_id: str,
        embedding_model_id: str,
        device: str,
        dtype: torch.dtype,
    ):
        self.summarizing_model_id = summarizing_model_id
        self.embedding_model_id = embedding_model_id
        self.device = device
        self.dtype = dtype
        self.embed_queue = deque()

        self.summarizing_model: Optional[Qwen3VLForConditionalGeneration] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.processor: Optional[AutoProcessor] = None

    async def load(self):
        """Load model + tokenizer if not already loaded."""
        if self.summarizing_model is not None and self.processor is not None:
            return

        start = time.perf_counter()
        logger.info(f"Loading processor and model for {self.summarizing_model_id}")
        self.processor = AutoProcessor.from_pretrained(
            self.summarizing_model_id, trust_remote_code=True
        )

        self.summarizing_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.summarizing_model_id,
            dtype=self.dtype,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).eval()

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Finished loading {self.summarizing_model_id} in {duration_ms:.2f} ms"
        )

        start = time.perf_counter()
        logger.info(f"Loading embedding model for {self.embedding_model_id}")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_id, device="cpu"
        )
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"Finished loading {self.embedding_model_id} in {duration_ms:.2f} ms"
        )

    async def unload(self):
        """Free model + tokenizer and clear CUDA cache."""
        if self.summarizing_model is not None:
            self.summarizing_model.to("cpu")
            del self.summarizing_model
            self.summarizing_model = None

        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def getSummarizer(self):
        """Return the summarizing model + processor or raise if not ready."""
        if self.summarizing_model is None or self.processor is None:
            raise ModelNotLoadedError("Summarizing model not loaded")
        return self.summarizing_model, self.processor

    def getEmbedder(self):
        """Return the embedding model or raise if not ready."""
        if self.embedding_model is None:
            raise ModelNotLoadedError("Embedding model not loaded")
        return self.embedding_model


model_manager = ModelManager(summarizing_model_id, embedding_model_id, DEVICE, DTYPE)


def generate_multimodal_summary(
    model, processor, text: str, images: Optional[list[str]] = None
) -> str:
    """Generate a multimodal summary given text and optional images."""
    human_message_content = []

    if images:
        for img_url in images:
            image_part = {"type": "image", "image": img_url}
            human_message_content.append(image_part)

    human_message_content = []

    # Add images first
    if images:
        for img_url in images:
            human_message_content.append({"type": "image", "image": img_url})

    # Build the text prompt
    prompt_text = f"""
    Text content: {text}

    Task: Create a brief, searchable summary (under 500 words total).

    Structure:
    **Overview:** 2 sentences - what this is about
    **Facts:** Bullet list - key details only  
    **Visual:** 2 sentences - image description
    **Questions:** List 4-5 questions (no answers)
    **Keywords:** 15-20 search terms

    Be concise and avoid repetition.

    Summary:
    """

    # Add the prompt as text
    human_message_content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": human_message_content}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=768, temperature=0.7, top_p=0.9, do_sample=True
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    summary = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return summary


async def process_embed_batches():
    """Process embedding requests in batches"""
    while True:
        if len(model_manager.embed_queue) >= BATCH_SIZE:
            await _process_embed_batch()
        elif len(model_manager.embed_queue) > 0:
            await asyncio.sleep(BATCH_TIMEOUT)
            if len(model_manager.embed_queue) > 0:
                await _process_embed_batch()
        else:
            await asyncio.sleep(0.01)


async def _process_embed_batch():
    """Batch embedding logic here..."""
    batch_size = min(BATCH_SIZE, len(model_manager.embed_queue))
    batch = [model_manager.embed_queue.popleft() for _ in range(batch_size)]

    texts = [req["text"] for req in batch]

    embedding_model = model_manager.getEmbedder()
    with torch.no_grad():
        embeddings = embedding_model.encode(texts)

    for i, req in enumerate(batch):
        req["future"].set_result(embeddings[i])


@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.load()
    asyncio.create_task(process_embed_batches())
    try:
        yield
    finally:
        await model_manager.unload()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    try:
        model_manager.getEmbedder()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"message": "API is running.", "device": str(DEVICE)}


@app.post("/summarize", response_model=SummarizationResponse)
def summarize(request: SummarizationRequest):
    model, processor = model_manager.getSummarizer()

    summary = generate_multimodal_summary(
        model, processor, request.chunk_text, request.image_urls
    )

    return SummarizationResponse(summary_text=summary)


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    """Create embedding for text"""
    future = asyncio.Future()
    model_manager.embed_queue.append(
        {"text": request.summarized_text, "future": future}
    )
    result = await future
    return {"embedding_vector": result.tolist()}
