from threading import BoundedSemaphore

from config import MAX_CONCURRENT_LLM_REQUESTS


llm_semaphore = BoundedSemaphore(
    MAX_CONCURRENT_LLM_REQUESTS
)


def try_acquire_llm_slot() -> bool:
    return llm_semaphore.acquire(blocking=False)


def release_llm_slot() -> None:
    llm_semaphore.release()