"""WebSocket fan-out for live pipeline events.

Architecture
------------

::

       +-----------------+     publish_event()     +---------------------+
       |  Celery / sync  | ----------------------> |   RunBroker         |
       |  training task  |                          |  per-run subscriber |
       +-----------------+                          |  asyncio.Queue list |
                                                     +----------+----------+
                                                                |
                                                                | fan-out
                                                                v
                                                     +---------------------+
                                                     |   WebSocket clients |
                                                     |  /ws/runs/{run_id}  |
                                                     +---------------------+

* Producers (Celery worker, fallback worker, in-process pipeline runner) call
  :func:`publish_event` from any thread; the broker thread-safely enqueues
  the payload into the per-connection :class:`asyncio.Queue`.
* Consumers (FastAPI WebSocket clients) connect to ``/ws/runs/{run_id}`` and
  receive a JSON stream of pipeline events as they arrive.

This is intentionally an in-memory broker. Production deployments that need
multi-replica fan-out should swap :class:`RunBroker` for a Redis Pub/Sub
implementation; the public ``publish_event`` / ``subscribe`` surface stays
the same so callers do not need to change.
"""

import asyncio
import json
import logging
import threading
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
except ImportError:
    APIRouter = None  # type: ignore[assignment,misc]
    WebSocket = None  # type: ignore[assignment,misc]

    class WebSocketDisconnect(Exception):  # type: ignore[no-redef]  # noqa: N818
        """Stand-in when FastAPI is not installed (name mirrors Starlette's)."""


logger = logging.getLogger(__name__)


class _Subscription:
    """Holds the event loop + queue for a single WebSocket connection."""

    __slots__ = ("run_id", "loop", "queue")

    def __init__(self, run_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self.run_id = run_id
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=1024)

    def deliver(self, event: Dict[str, Any]) -> None:
        """Schedule an event onto this subscriber's loop."""
        try:
            self.loop.call_soon_threadsafe(self._enqueue, event)
        except RuntimeError:
            logger.debug("Subscription loop is no longer running; dropping event.")

    def _enqueue(self, event: Dict[str, Any]) -> None:
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self.queue.put_nowait(event)


class RunBroker:
    """Thread-safe pub/sub for per-run event streams.

    The broker keeps one :class:`asyncio.Queue` per WebSocket subscriber and
    fans every published event out to every queue. It captures the event
    loop at subscription time so producers can publish from any thread
    (notably Celery workers, which are usually not on the same loop).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: Dict[str, List[_Subscription]] = {}

    def subscribe(self, run_id: str) -> _Subscription:
        """Register a new subscriber for ``run_id``."""
        sub = _Subscription(run_id=run_id, loop=asyncio.get_event_loop())
        with self._lock:
            self._subs.setdefault(run_id, []).append(sub)
        return sub

    def unsubscribe(self, sub: _Subscription) -> None:
        """Drop a subscriber from the broker."""
        with self._lock:
            subs = self._subs.get(sub.run_id, [])
            try:
                subs.remove(sub)
            except ValueError:
                pass
            if not subs:
                self._subs.pop(sub.run_id, None)

    def publish(self, run_id: str, event: Dict[str, Any]) -> int:
        """Fan an event out to all subscribers for ``run_id``.

        Returns the number of subscribers that received the event. Producers
        do not need to await — this method is safe to call from any thread.
        """
        with self._lock:
            subs = list(self._subs.get(run_id, []))
        for sub in subs:
            sub.deliver(event)
        return len(subs)


_BROKER = RunBroker()


def get_broker() -> RunBroker:
    """Return the process-wide broker singleton."""
    return _BROKER


def publish_event(run_id: str, event: Dict[str, Any]) -> int:
    """Convenience wrapper for :meth:`RunBroker.publish`."""
    return _BROKER.publish(run_id, event)


def build_realtime_router() -> Optional[Any]:
    """Construct the WebSocket router or ``None`` if FastAPI is missing."""
    if APIRouter is None:
        return None

    router = APIRouter(prefix="/ws", tags=["realtime"])

    @router.websocket("/runs/{run_id}")
    async def stream_run_events(websocket: WebSocket, run_id: str) -> None:
        await websocket.accept()
        sub = _BROKER.subscribe(run_id)
        await websocket.send_text(json.dumps({"type": "subscribed", "run_id": run_id}))
        try:
            while True:
                event = await sub.queue.get()
                await websocket.send_text(json.dumps(event, default=str))
                if event.get("type") in {"completed", "error"}:
                    break
        except WebSocketDisconnect:
            logger.debug("WebSocket client for run %s disconnected.", run_id)
        except Exception:
            logger.exception("WebSocket stream for run %s errored.", run_id)
        finally:
            _BROKER.unsubscribe(sub)
            try:
                await websocket.close()
            except Exception:
                pass

    return router
