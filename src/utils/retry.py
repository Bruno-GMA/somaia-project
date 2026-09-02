import asyncio
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """
    Execute function with exponential backoff retry.
    """
    last_exception: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                if jitter:
                    delay *= (0.5 + random.random())
                print(f"  Tentativa {attempt + 1} falhou: {e}. Retentando em {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"  Todas as {max_retries + 1} tentativas falharam.")
    assert last_exception is not None
    raise last_exception


async def async_retry_with_backoff(
    func: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Any:
    """
    Execute async function with exponential backoff retry.
    """
    last_exception: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (exponential_base ** attempt), max_delay)
                if jitter:
                    delay *= (0.5 + random.random())
                print(f"  Tentativa {attempt + 1} falhou: {e}. Retentando em {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                print(f"  Todas as {max_retries + 1} tentativas falharam.")
    assert last_exception is not None
    raise last_exception
