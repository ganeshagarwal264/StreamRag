import asyncio
import time
import os
import logging

class TokenBucketRateLimiter:
    def __init__(self):
        self.capacity = int(os.environ.get("RPM_LIMIT", 5))
        self.tokens = float(self.capacity)
        self.refill_rate = self.capacity / 60.0 if self.capacity > 0 else 0
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        if self.capacity <= 0:
            return # disabled
        async with self.lock:
            while True:
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)
                self.last_update = now
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                else:
                    wait_time = (1.0 - self.tokens) / self.refill_rate
                    logging.info(f"RateLimiter: waiting {wait_time:.2f}s for token")
                    await asyncio.sleep(wait_time)

rate_limiter = TokenBucketRateLimiter()
