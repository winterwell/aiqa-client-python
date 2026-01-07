"""
OpenTelemetry span exporter that sends spans to the AIQA server API.
Buffers spans and flushes them periodically or on shutdown. Thread-safe.
"""

import os
import json
import logging
import threading
import time
import io
import asyncio
from typing import List, Dict, Any, Optional
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .constants import AIQA_TRACER_NAME, VERSION, LOG_TAG
from .http_utils import get_server_url, get_api_key, build_headers
from .object_serialiser import toNumber, safe_json_dumps

logger = logging.getLogger(LOG_TAG)


class AIQASpanExporter(SpanExporter):
    """
    Exports spans to AIQA server. Buffers spans and auto-flushes every flush_interval_seconds.
    Call shutdown() before process exit to flush remaining spans.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        flush_interval_seconds: float = 5.0,
        max_batch_size_bytes: int = 5 * 1024 * 1024,  # 5MB default
        max_buffer_spans: Optional[int] = None,  # Maximum spans to buffer (prevents unbounded growth)
        max_buffer_size_bytes: Optional[int] = None,  # Maximum buffer size in bytes (prevents unbounded memory growth)
        startup_delay_seconds: Optional[float] = None,
    ):
        """
        Initialize the AIQA span exporter.

        Args:
            server_url: URL of the AIQA server (defaults to AIQA_SERVER_URL env var or https://server-aiqa.winterwell.com)
            api_key: API key for authentication (defaults to AIQA_API_KEY env var)
            flush_interval_seconds: How often to flush spans to the server
            max_batch_size_bytes: Maximum size of a single batch in bytes (default: 5mb)
            max_buffer_spans: Maximum spans to buffer (prevents unbounded growth). 
                Defaults to 10000, or AIQA_MAX_BUFFER_SPANS env var if set.
            max_buffer_size_bytes: Maximum total buffer size in bytes (prevents unbounded memory growth).
                Defaults to None (no limit), or AIQA_MAX_BUFFER_SIZE_BYTES env var if set.
            startup_delay_seconds: Delay before starting auto-flush (default: 10s, or AIQA_STARTUP_DELAY_SECONDS env var)
        """
        self._server_url = get_server_url(server_url)
        self._api_key = get_api_key(api_key)
        self.flush_interval_ms = flush_interval_seconds * 1000
        self.max_batch_size_bytes = max_batch_size_bytes
        
        # Get max_buffer_spans from parameter, environment variable, or default
        if not max_buffer_spans:
            max_buffer_spans = toNumber(os.getenv("AIQA_MAX_BUFFER_SPANS")) or 10000
        self.max_buffer_spans = max_buffer_spans
        
        # Get max_buffer_size_bytes from parameter, environment variable, or default
        if not max_buffer_size_bytes:
            max_buffer_size_bytes = toNumber(os.getenv("AIQA_MAX_BUFFER_SIZE_BYTES")) or toNumber("100m")
        self.max_buffer_size_bytes = max_buffer_size_bytes
        
        # Get startup delay from parameter or environment variable (default: 10s)
        if startup_delay_seconds is None:
            env_delay = os.getenv("AIQA_STARTUP_DELAY_SECONDS")
            if env_delay:
                try:
                    startup_delay_seconds = float(env_delay)
                except ValueError:
                    logger.warning(f"Invalid AIQA_STARTUP_DELAY_SECONDS value '{env_delay}', using default 10.0")
                    startup_delay_seconds = 10.0
            else:
                startup_delay_seconds = 10.0
        self.startup_delay_seconds = startup_delay_seconds
        
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_span_keys: set = set()  # Track (traceId, spanId) tuples to prevent duplicates (Python 3.8 compatible)
        self.buffer_size_bytes: int = 0  # Track total size of buffered spans in bytes
        # Cache span sizes to avoid recalculation (maps span_key -> size_bytes)
        # Limited to max_buffer_spans * 2 to prevent unbounded growth
        self._span_size_cache: Dict[tuple, int] = {}
        self._max_cache_size = self.max_buffer_spans * 2  # Allow cache to be 2x buffer size
        self.buffer_lock = threading.Lock()
        self.flush_lock = threading.Lock()
        # shutdown_requested is only set once (in shutdown()) and read many times
        # No lock needed: worst case is reading stale False, which is acceptable
        self.shutdown_requested = False
        self.flush_timer: Optional[threading.Thread] = None
        self._auto_flush_started = False
        self._auto_flush_lock = threading.Lock()  # Lock for lazy thread creation
        
        logger.info(f"Initializing AIQASpanExporter: server_url={self._server_url or 'not set'}, "
            f"flush_interval={flush_interval_seconds}s, startup_delay={startup_delay_seconds}s"
        )
        # Don't start thread immediately - start lazily on first export to avoid startup issues

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        """
        Export spans to the AIQA server. Adds spans to buffer for async flushing.
        Deduplicates spans based on (traceId, spanId) to prevent repeated exports.
        Actual send is done by flush -> _send_spans, or shutdown -> _send_spans_sync
        """
        if not spans:
            logger.debug(f"export: called with empty spans list")
            return SpanExportResult.SUCCESS
        
        # Check if AIQA tracing is enabled
        try:
            from .client import get_aiqa_client
            client = get_aiqa_client()
            if not client.enabled:
                logger.debug(f"AIQA export: skipped: tracing is disabled")
                return SpanExportResult.SUCCESS
        except Exception:
            # If we can't check enabled status, proceed (fail open)
            pass
        
        logger.debug(f"AIQA export() to buffer called with {len(spans)} spans")
        
        # Lazy initialization: start auto-flush thread on first export
        # This avoids thread creation during initialization, which can cause issues in ECS deployments
        self._ensure_auto_flush_started()
        
        # Serialize and add to buffer, deduplicating by (traceId, spanId)
        with self.buffer_lock:
            serialized_spans = []
            serialized_sizes = []  # Track sizes of serialized spans
            duplicates_count = 0
            dropped_count = 0
            dropped_memory_count = 0
            flush_in_progress = self.flush_lock.locked()
            
            for span in spans:
                # Check if buffer is full by span count (prevent unbounded growth)
                if len(self.buffer) >= self.max_buffer_spans:
                    if flush_in_progress:
                        # Flush in progress, drop this span
                        dropped_count += 1
                        continue
                    # Flush not in progress, will trigger flush after adding spans
                    # Continue processing remaining spans to add them before flush
                
                serialized = self._serialize_span(span)
                span_key = (serialized["traceId"], serialized["spanId"])
                if span_key not in self.buffer_span_keys:
                    # Estimate size of this span when serialized (cache for later use)
                    span_size = self._get_span_size(span_key, serialized)
                    
                    # Check if buffer is full by memory size (prevent unbounded memory growth)
                    if self.max_buffer_size_bytes is not None and self.buffer_size_bytes + span_size > self.max_buffer_size_bytes:
                        if flush_in_progress:
                            # Flush in progress, drop this span
                            # Don't cache size for dropped spans to prevent memory leak
                            dropped_memory_count += 1
                            continue
                        # Flush not in progress, will trigger flush after adding spans
                        # Continue processing remaining spans to add them before flush
                    
                    serialized_spans.append(serialized)
                    serialized_sizes.append(span_size)
                    self.buffer_span_keys.add(span_key)
                else:
                    duplicates_count += 1
                    logger.debug(f"export: skipping duplicate span: traceId={serialized['traceId']}, spanId={serialized['spanId']}")
            
            # Add spans and update buffer size
            self.buffer.extend(serialized_spans)
            self.buffer_size_bytes += sum(serialized_sizes)
            buffer_size = len(self.buffer)
            
            # Check if thresholds are reached after adding spans
            threshold_reached = self._check_thresholds_reached()
            
            if dropped_count > 0:
                logger.warning(f"WARNING: Buffer full ({buffer_size} spans), dropped {dropped_count} span(s) (flush in progress). "
                    f"Consider increasing max_buffer_spans or fixing server connectivity."
                )
            if dropped_memory_count > 0:
                logger.warning(f"WARNING: Buffer memory limit reached ({self.buffer_size_bytes} bytes / {self.max_buffer_size_bytes} bytes), "
                    f"dropped {dropped_memory_count} span(s) (flush in progress). "
                    f"Consider increasing AIQA_MAX_BUFFER_SIZE_BYTES or fixing server connectivity."
                )
        
        # Trigger immediate flush if threshold reached and flush not in progress
        if threshold_reached and not flush_in_progress:
            logger.info(f"Buffer threshold reached ({buffer_size} spans, {self.buffer_size_bytes} bytes), triggering immediate flush")
            self._trigger_immediate_flush()
        
        if duplicates_count > 0:
            logger.debug(f"export() added {len(serialized_spans)} span(s) to buffer, skipped {duplicates_count} duplicate(s). "
                f"Total buffered: {buffer_size}"
            )
        else:
            logger.debug(f"export() added {len(spans)} span(s) to buffer. "
                f"Total buffered: {buffer_size}"
            )

        return SpanExportResult.SUCCESS

    def _serialize_span(self, span: ReadableSpan) -> Dict[str, Any]:
        """Convert ReadableSpan to a serializable format."""
        span_context = span.get_span_context()
        
        # Get parent span ID
        parent_span_id = None
        if hasattr(span, "parent") and span.parent:
            parent_span_id = format(span.parent.span_id, "016x")
        elif hasattr(span, "parent_span_id") and span.parent_span_id:
            parent_span_id = format(span.parent_span_id, "016x")
        
        # Get span kind (handle both enum and int)
        span_kind = span.kind
        if hasattr(span_kind, "value"):
            span_kind = span_kind.value
        
        # Get status code (handle both enum and int)
        status_code = span.status.status_code
        if hasattr(status_code, "value"):
            status_code = status_code.value
        
        return {
            "name": span.name,
            "kind": span_kind,
            "parentSpanId": parent_span_id,
            "startTime": self._time_to_tuple(span.start_time),
            "endTime": self._time_to_tuple(span.end_time) if span.end_time else None,
            "status": {
                "code": status_code,
                "message": getattr(span.status, "description", None),
            },
            "attributes": dict(span.attributes) if span.attributes else {},
            "links": [
                {
                    "context": {
                        "traceId": format(link.context.trace_id, "032x"),
                        "spanId": format(link.context.span_id, "016x"),
                    },
                    "attributes": dict(link.attributes) if link.attributes else {},
                }
                for link in (span.links or [])
            ],
            "events": [
                {
                    "name": event.name,
                    "time": self._time_to_tuple(event.timestamp),
                    "attributes": dict(event.attributes) if event.attributes else {},
                }
                for event in (span.events or [])
            ],
            "resource": {
                "attributes": dict(span.resource.attributes) if span.resource.attributes else {},
            },
            "traceId": format(span_context.trace_id, "032x"),
            "spanId": format(span_context.span_id, "016x"),
            "traceFlags": span_context.trace_flags,
            "duration": self._time_to_tuple(span.end_time - span.start_time) if span.end_time else None,
            "ended": span.end_time is not None,
            "instrumentationLibrary": self._get_instrumentation_library(span),
        }

    def _get_instrumentation_library(self, span: ReadableSpan) -> Dict[str, Any]:
        """
        Get instrumentation library information from the span: just use the package version.
        """
        return {
            "name": AIQA_TRACER_NAME,
            "version": VERSION,
        }

    def _time_to_tuple(self, nanoseconds: int) -> tuple:
        """Convert nanoseconds to (seconds, nanoseconds) tuple."""
        seconds = int(nanoseconds // 1_000_000_000)
        nanos = int(nanoseconds % 1_000_000_000)
        return (seconds, nanos)

    def _get_span_size(self, span_key: tuple, serialized: Dict[str, Any]) -> int:
        """
        Get span size from cache or calculate and cache it.
        Thread-safe when called within buffer_lock.
        Limits cache size to prevent unbounded memory growth.
        """
        if span_key in self._span_size_cache:
            return self._span_size_cache[span_key]
        span_json = safe_json_dumps(serialized)
        span_size = len(span_json.encode('utf-8'))
        # Only cache if we have valid keys and cache isn't too large
        if span_key[0] and span_key[1] and len(self._span_size_cache) < self._max_cache_size:
            self._span_size_cache[span_key] = span_size
        return span_size

    def _check_thresholds_reached(self) -> bool:
        """Check if buffer thresholds are reached. Must be called within buffer_lock."""
        if len(self.buffer) >= self.max_buffer_spans:
            return True
        if self.max_buffer_size_bytes is not None and self.buffer_size_bytes >= self.max_buffer_size_bytes:
            return True
        return False

    def _build_request_headers(self) -> Dict[str, str]:
        """Build HTTP headers for span requests."""
        return build_headers(self._api_key)

    def _get_span_url(self) -> str:
        return f"{self._server_url}/span"

    def _is_interpreter_shutdown_error(self, error: Exception) -> bool:
        """Check if error is due to interpreter shutdown."""
        error_str = str(error)
        return "cannot schedule new futures after" in error_str or "interpreter shutdown" in error_str

    def _extract_spans_from_buffer(self) -> List[Dict[str, Any]]:
        """Extract spans from buffer (thread-safe). Returns copy of buffer."""
        with self.buffer_lock:
            return self.buffer[:]

    def _extract_and_remove_spans_from_buffer(self) -> List[Dict[str, Any]]:
        """
        Atomically extract and remove all spans from buffer (thread-safe).
        Returns the extracted spans. This prevents race conditions where spans
        are added between extraction and clearing.
        Note: Does NOT clear buffer_span_keys - that should be done after successful send
        to avoid unnecessary clearing/rebuilding on failures.
        Also resets buffer_size_bytes to 0.
        """
        with self.buffer_lock:
            spans = self.buffer[:]
            self.buffer.clear()
            self.buffer_size_bytes = 0
            return spans
    
    def _remove_span_keys_from_tracking(self, spans: List[Dict[str, Any]]) -> None:
        """
        Remove span keys from tracking set and size cache (thread-safe). Called after successful send.
        """
        with self.buffer_lock:
            for span in spans:
                span_key = (span["traceId"], span["spanId"])
                self.buffer_span_keys.discard(span_key)
                # Also remove from size cache to free memory
                self._span_size_cache.pop(span_key, None)

    def _prepend_spans_to_buffer(self, spans: List[Dict[str, Any]]) -> None:
        """
        Prepend spans back to buffer (thread-safe). Used to restore spans
        if sending fails. Rebuilds the span keys tracking set and buffer size.
        Uses cached sizes when available to avoid re-serialization.
        """
        with self.buffer_lock:
            self.buffer[:0] = spans
            # Rebuild span keys set from current buffer contents
            self.buffer_span_keys = {(span["traceId"], span["spanId"]) for span in self.buffer}
            # Recalculate buffer size using cache when available
            total_size = 0
            for span in self.buffer:
                span_key = (span.get("traceId"), span.get("spanId"))
                total_size += self._get_span_size(span_key, span)
            self.buffer_size_bytes = total_size

    def _clear_buffer(self) -> None:
        """Clear the buffer (thread-safe)."""
        with self.buffer_lock:
            self.buffer.clear()
            self.buffer_span_keys.clear()
            self.buffer_size_bytes = 0
            self._span_size_cache.clear()

    def _split_into_batches(self, spans: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split spans into batches based on max_batch_size_bytes.
        Each batch will be as large as possible without exceeding the limit.
        If a single span exceeds the limit, it will be sent in its own batch with a warning.
        """
        if not spans:
            return []
        
        batches = []
        current_batch = []
        current_batch_size = 0
        
        for span in spans:
            # Get size from cache if available, otherwise calculate it
            span_key = (span.get("traceId"), span.get("spanId"))
            span_size = self._get_span_size(span_key, span)
            
            # Check if this single span exceeds the limit
            if span_size > self.max_batch_size_bytes:
                # If we have a current batch, save it first
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_size = 0
                
                # Log warning about oversized span
                span_name = span.get('name', 'unknown')
                span_trace_id = span.get('traceId', 'unknown')
                logger.warning(f"Span \'{span_name}' (traceId={span_trace_id}) exceeds max_batch_size_bytes "
                    f"({span_size} bytes > {self.max_batch_size_bytes} bytes). "
                    f"Will attempt to send it anyway - may fail if server/nginx limit is exceeded."
                )
                # Still create a batch with just this span - we'll try to send it
                batches.append([span])
                continue
            
            # If adding this span would exceed the limit, start a new batch
            if current_batch and current_batch_size + span_size > self.max_batch_size_bytes:
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0
            
            current_batch.append(span)
            current_batch_size += span_size
        
        # Add the last batch if it has any spans
        if current_batch:
            batches.append(current_batch)
        
        return batches

    async def flush(self) -> None:
        """
        Flush buffered spans to the server. Thread-safe: ensures only one flush operation runs at a time.
        Atomically extracts spans to prevent race conditions with concurrent export() calls.
        
        Lock ordering: flush_lock -> buffer_lock (must be consistent to avoid deadlocks)
        """
        logger.debug(f"flush: called - attempting to acquire flush lock")
        with self.flush_lock:
            logger.debug(f"flush() acquired flush lock")
            # Atomically extract and remove spans to prevent race conditions
            # where export() adds spans between extraction and clearing
            spans_to_flush = self._extract_and_remove_spans_from_buffer()
            logger.debug(f"flush: extracted {len(spans_to_flush)} span(s) from buffer")

            if not spans_to_flush:
                logger.debug(f"flush() completed: no spans to flush")
                return

            # Skip sending if server URL is not configured
            if not self._server_url:
                logger.warning(f"Skipping flush: AIQA_SERVER_URL is not set. {len(spans_to_flush)} span(s) will not be sent."
                )
                # Spans already removed from buffer, clear their keys to free memory
                self._remove_span_keys_from_tracking(spans_to_flush)
                return

        # Release flush_lock before I/O to avoid blocking other flush attempts
        # Spans are already extracted, so concurrent exports won't interfere
        logger.info(f"flush: sending {len(spans_to_flush)} span(s) to server")
        try:
            await self._send_spans(spans_to_flush)
            logger.info(f"flush() successfully sent {len(spans_to_flush)} span(s) to server")
            # Spans already removed from buffer during extraction
            # Now clear their keys from tracking set to free memory
            self._remove_span_keys_from_tracking(spans_to_flush)
        except RuntimeError as error:
            if self._is_interpreter_shutdown_error(error):
                if self.shutdown_requested:
                    logger.debug(f"flush: skipped due to interpreter shutdown: {error}")
                else:
                    logger.warning(f"flush() interrupted by interpreter shutdown: {error}")
                # Put spans back for retry with sync send during shutdown
                self._prepend_spans_to_buffer(spans_to_flush)
                raise
            logger.error(f"Error flushing spans to server: {error}")
            # Put spans back for retry
            self._prepend_spans_to_buffer(spans_to_flush)
            raise
        except Exception as error:
            logger.error(f"Error flushing spans to server: {error}")
            # Put spans back for retry
            self._prepend_spans_to_buffer(spans_to_flush)
            if self.shutdown_requested:
                raise

    def _ensure_auto_flush_started(self) -> None:
        """Ensure auto-flush thread is started (lazy initialization). Thread-safe."""
        # Fast path: check without lock first
        if self._auto_flush_started or self.shutdown_requested:
            return
        
        # Slow path: acquire lock and double-check
        with self._auto_flush_lock:
            if self._auto_flush_started or self.shutdown_requested:
                return
            
            try:
                self._start_auto_flush()
                self._auto_flush_started = True
            except Exception as e:
                logger.error(f"Failed to start auto-flush thread: {e}", exc_info=True)
                # Don't raise - allow spans to be buffered even if auto-flush fails
                # They can still be flushed manually or on shutdown

    def _trigger_immediate_flush(self) -> None:
        """
        Trigger an immediate flush in a background thread.
        This is called when buffer thresholds are reached and no flush is in progress.
        """
        def flush_in_thread():
            """Run flush in a new thread with its own event loop."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.flush())
                finally:
                    if not loop.is_closed():
                        loop.close()
            except Exception as e:
                logger.error(f"Error in immediate flush thread: {e}", exc_info=True)
        
        # Start flush in background thread (daemon so it doesn't block shutdown)
        flush_thread = threading.Thread(target=flush_in_thread, daemon=True, name="AIQA-ImmediateFlush")
        flush_thread.start()

    def _flush_worker(self) -> None:
        """Worker function for auto-flush thread. Runs in a separate thread with its own event loop."""
        import asyncio
        logger.debug(f"Auto-flush worker thread started")
        
        # Wait for startup delay before beginning flush operations
        # This gives the container/application time to stabilize, which helps avoid startup issues (seen with AWS ECS, Dec 2025).
        if self.startup_delay_seconds > 0:
            logger.info(f"Auto-flush waiting {self.startup_delay_seconds}s before first flush (startup delay)")
            # Sleep in small increments to allow for early shutdown
            sleep_interval = 0.5
            remaining_delay = self.startup_delay_seconds
            while remaining_delay > 0 and not self.shutdown_requested:
                sleep_time = min(sleep_interval, remaining_delay)
                time.sleep(sleep_time)
                remaining_delay -= sleep_time
            
            if self.shutdown_requested:
                logger.debug(f"Auto-flush startup delay interrupted by shutdown")
                return
            
            logger.info(f"Auto-flush startup delay complete, beginning flush operations")
        
        # Create event loop in this thread (isolated from main thread's event loop)
        # This prevents interference with the main application's event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception as e:
            logger.error(f"Failed to create event loop for auto-flush thread: {e}", exc_info=True)
            return
        
        # Ensure event loop is always closed, even if an exception occurs
        try:
            cycle_count = 0
            while not self.shutdown_requested:
                cycle_count += 1
                logger.debug(f"Auto-flush cycle #{cycle_count} starting")
                try:
                    loop.run_until_complete(self.flush())
                    logger.debug(f"Auto-flush cycle #{cycle_count} completed, sleeping {self.flush_interval_ms / 1000.0}s")
                except Exception as e:
                    logger.error(f"Error in auto-flush cycle #{cycle_count}: {e}")
                    logger.debug(f"Auto-flush cycle #{cycle_count} error handled, sleeping {self.flush_interval_ms / 1000.0}s")
                
                # Sleep after each cycle (including errors) to avoid tight loops
                if not self.shutdown_requested:
                    time.sleep(self.flush_interval_ms / 1000.0)
            
            logger.info(f"Auto-flush worker thread stopping (shutdown requested). Completed {cycle_count} cycles.")
            # Don't do final flush here - shutdown() will handle it with synchronous send
            # This avoids event loop shutdown issues
            logger.debug(f"Auto-flush thread skipping final flush (will be handled by shutdown() with sync send)")
        finally:
            # Always close the event loop, even if an exception occurs
            try:
                if not loop.is_closed():
                    loop.close()
                logger.debug(f"Auto-flush worker thread event loop closed")
            except Exception:
                pass  # Ignore errors during cleanup
    
    def _start_auto_flush(self) -> None:
        """Start the auto-flush timer with startup delay."""
        if self.shutdown_requested:
            logger.warning(f"_start_auto_flush() called but shutdown already requested")
            return

        logger.info(f"Starting auto-flush thread with interval {self.flush_interval_ms / 1000.0}s, "
            f"startup delay {self.startup_delay_seconds}s"
        )

        flush_thread = threading.Thread(target=self._flush_worker, daemon=True, name="AIQA-AutoFlush")
        flush_thread.start()
        self.flush_timer = flush_thread
        logger.info(f"Auto-flush thread started: {flush_thread.name} (daemon={flush_thread.daemon})")

    async def _send_spans(self, spans: List[Dict[str, Any]]) -> None:
        """Send spans to the server API (async). Batches large payloads automatically."""
        import aiohttp

        # Split into batches if needed
        batches = self._split_into_batches(spans)
        if len(batches) > 1:
            logger.info(f"_send_spans: splitting {len(spans)} spans into {len(batches)} batches")
        
        url = self._get_span_url()
        headers = self._build_request_headers()
        
        if not self._api_key: # This should not happen
            logger.error(f"_send_spans: fail - no API key provided. {len(spans)} spans lost.")
            # Spans were already removed from buffer before calling this method. They will now get forgotten
            return

        # Use timeout to prevent hanging on unreachable servers
        timeout = aiohttp.ClientTimeout(total=30.0, connect=10.0)
        errors = []
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for batch_idx, batch in enumerate(batches):
                try:
                    logger.debug(f"_send_spans: sending batch {batch_idx + 1}/{len(batches)} with {len(batch)} spans to {url}")
                    # Pre-serialize JSON to bytes and wrap in BytesIO to avoid blocking event loop
                    json_bytes = json.dumps(batch).encode('utf-8')
                    data = io.BytesIO(json_bytes)
                    
                    async with session.post(url, data=data, headers=headers) as response:
                        logger.debug(f"_send_spans: batch {batch_idx + 1} received response: status={response.status}")
                        if not response.ok:
                            error_text = await response.text()
                            error_msg = f"Failed to send batch {batch_idx + 1}/{len(batches)}: {response.status} {response.reason} - {error_text[:200]}"
                            logger.error(f"_send_spans: {error_msg}")
                            errors.append((batch_idx + 1, error_msg))
                            # Continue with other batches even if one fails
                            continue
                        logger.debug(f"_send_spans: batch {batch_idx + 1} successfully sent {len(batch)} spans")
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    # Network errors and timeouts - log but don't fail completely
                    error_msg = f"Network error in batch {batch_idx + 1}: {type(e).__name__}: {e}"
                    logger.warning(f"_send_spans: {error_msg} - will retry on next flush")
                    errors.append((batch_idx + 1, error_msg))
                    # Continue with other batches
                except RuntimeError as e:
                    if self._is_interpreter_shutdown_error(e):
                        if self.shutdown_requested:
                            logger.debug(f"_send_spans: skipped due to interpreter shutdown: {e}")
                        else:
                            logger.warning(f"_send_spans: interrupted by interpreter shutdown: {e}")
                        raise
                    error_msg = f"RuntimeError in batch {batch_idx + 1}: {type(e).__name__}: {e}"
                    logger.error(f"_send_spans: {error_msg}")
                    errors.append((batch_idx + 1, error_msg))
                    # Continue with other batches
                except Exception as e:
                    error_msg = f"Exception in batch {batch_idx + 1}: {type(e).__name__}: {e}"
                    logger.error(f"_send_spans: {error_msg}")
                    errors.append((batch_idx + 1, error_msg))
                    # Continue with other batches
        
        # If any batches failed, raise an exception with details
        # Spans will be restored to buffer for retry on next flush
        if errors:
            error_summary = "; ".join([f"batch {idx}: {msg}" for idx, msg in errors])
            raise Exception(f"Failed to send some spans: {error_summary}")
        
        logger.debug(f"_send_spans: successfully sent all {len(spans)} spans in {len(batches)} batch(es)")

    def _send_spans_sync(self, spans: List[Dict[str, Any]]) -> None:
        """Send spans to the server API (synchronous, for shutdown scenarios). Batches large payloads automatically."""
        import requests

        # Split into batches if needed
        batches = self._split_into_batches(spans)
        if len(batches) > 1:
            logger.info(f"_send_spans_sync() splitting {len(spans)} spans into {len(batches)} batches")
        
        url = self._get_span_url()
        headers = self._build_request_headers()
        
        if not self._api_key:
            logger.error(f"_send_spans_sync() fail - no API key provided")
            return

        errors = []
        for batch_idx, batch in enumerate(batches):
            try:
                logger.debug(f"_send_spans_sync() sending batch {batch_idx + 1}/{len(batches)} with {len(batch)} spans to {url}")
                response = requests.post(url, json=batch, headers=headers, timeout=10.0)
                logger.debug(f"_send_spans_sync() batch {batch_idx + 1} received response: status={response.status_code}")
                if not response.ok:
                    error_text = response.text[:200] if response.text else ""
                    error_msg = f"Failed to send batch {batch_idx + 1}/{len(batches)}: {response.status_code} {response.reason} - {error_text}"
                    logger.error(f"_send_spans_sync() {error_msg}")
                    errors.append((batch_idx + 1, error_msg))
                    # Continue with other batches even if one fails
                    continue
                logger.debug(f"_send_spans_sync() batch {batch_idx + 1} successfully sent {len(batch)} spans")
            except Exception as e:
                error_msg = f"Exception in batch {batch_idx + 1}: {type(e).__name__}: {e}"
                logger.error(f"_send_spans_sync() {error_msg}")
                errors.append((batch_idx + 1, error_msg))
                # Continue with other batches
        
        # If any batches failed, raise an exception with details
        if errors:
            error_summary = "; ".join([f"batch {idx}: {msg}" for idx, msg in errors])
            raise Exception(f"Failed to send some spans: {error_summary}")
        
        logger.debug(f"_send_spans_sync() successfully sent all {len(spans)} spans in {len(batches)} batch(es)")

    def shutdown(self) -> None:
        """Shutdown the exporter, flushing any remaining spans. Call before process exit."""
        logger.info(f"shutdown: called - initiating exporter shutdown")
        self.shutdown_requested = True

        # Check buffer state before shutdown
        with self.buffer_lock:
            buffer_size = len(self.buffer)
            logger.info(f"shutdown: buffer contains {buffer_size} span(s) before shutdown")

        # Wait for flush thread to finish (it will do final flush)
        # Only wait if thread was actually started
        if self._auto_flush_started and self.flush_timer and self.flush_timer.is_alive():
            logger.info(f"shutdown: waiting for auto-flush thread to complete (timeout=10s)")
            self.flush_timer.join(timeout=10.0)
            if self.flush_timer.is_alive():
                logger.warning(f"shutdown: auto-flush thread did not complete within timeout")
            else:
                logger.info(f"shutdown: auto-flush thread completed")
        else:
            logger.debug(f"shutdown: no active auto-flush thread to wait for")

        # Final flush attempt (use synchronous send to avoid event loop issues)
        with self.flush_lock:
            logger.debug(f"shutdown: performing final flush with synchronous send")
            # Atomically extract and remove spans to prevent race conditions
            spans_to_flush = self._extract_and_remove_spans_from_buffer()
            logger.debug(f"shutdown: extracted {len(spans_to_flush)} span(s) from buffer for final flush")

            if spans_to_flush:
                if not self._server_url:
                    logger.warning(f"shutdown: skipping final flush: AIQA_SERVER_URL is not set. "
                        f"{len(spans_to_flush)} span(s) will not be sent."
                    )
                    # Spans already removed from buffer, clear their keys to free memory
                    self._remove_span_keys_from_tracking(spans_to_flush)
                else:
                    logger.info(f"shutdown: sending {len(spans_to_flush)} span(s) to server (synchronous)")
                    try:
                        self._send_spans_sync(spans_to_flush)
                        logger.info(f"shutdown: successfully sent {len(spans_to_flush)} span(s) to server")
                        # Spans already removed from buffer during extraction
                        # Clear their keys from tracking set to free memory
                        self._remove_span_keys_from_tracking(spans_to_flush)
                    except Exception as e:
                        logger.error(f"shutdown: failed to send spans: {e}")
                        # Spans already removed, but process is exiting anyway
                        logger.warning(f"shutdown: {len(spans_to_flush)} span(s) were not sent due to error")
                        # Keys will remain in tracking set, but process is exiting so memory will be freed
            else:
                logger.debug(f"shutdown: no spans to flush")
        
        # Check buffer state after shutdown
        with self.buffer_lock:
            buffer_size = len(self.buffer)
            if buffer_size > 0:
                logger.warning(f"shutdown: buffer still contains {buffer_size} span(s) after shutdown")
            else:
                logger.info(f"shutdown: buffer is empty after shutdown")
        
        logger.info(f"shutdown: completed")

