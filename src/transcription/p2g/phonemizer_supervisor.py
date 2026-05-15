import logging
import multiprocessing as mp
from pathlib import Path
import queue
import time
from collections import Counter
from typing import Optional


logger = logging.getLogger(__name__)

_REQUEST_SINGLE = "single"
_REQUEST_BATCH = "batch"
_REQUEST_STOP = "stop"
_STATUS_OK = "ok"
_STATUS_TRANSFORM_ERROR = "transform_error"
_STATUS_CHILD_CRASH = "child_crash"
_STATUS_TIMEOUT = "timeout"
_STATUS_MONITOR_ERROR = "monitor_error"
_STATUS_MONITOR_EXIT = "monitor_exit"
_POLL_INTERVAL_SECONDS = 0.2
_RESULT_GRACE_SECONDS = 5.0


def _phonemizer_child_main(
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        espeak_path: Optional[str],
) -> None:
    from transcription.g2p import phonemize_romanize

    if espeak_path is not None:
        phonemize_romanize.CustomEspeakBackend.set_library(espeak_path)
        phonemize_romanize.BACKENDS.clear()

    while True:
        message = input_queue.get()
        request_type = message[0]
        if request_type == _REQUEST_STOP:
            return

        _, request_id, payload, language = message
        try:
            if request_type == _REQUEST_SINGLE:
                result = phonemize_romanize.strip_ipa(
                    phonemize_romanize.phonemize_batch([payload], language)[0]
                )
            elif request_type == _REQUEST_BATCH:
                result = [
                    phonemize_romanize.strip_ipa(item)
                    for item in phonemize_romanize.phonemize_batch(payload, language)
                ]
            else:
                raise ValueError(f"unsupported request type: {request_type}")
            output_queue.put((_STATUS_OK, request_id, result))
        except Exception as exc:
            output_queue.put((_STATUS_TRANSFORM_ERROR, request_id, f"{type(exc).__name__}: {exc}"))


def _close_queue(work_queue: Optional[mp.Queue]) -> None:
    if work_queue is None:
        return
    try:
        work_queue.close()
    except Exception:
        pass
    try:
        work_queue.join_thread()
    except Exception:
        pass


def _stop_process(process: Optional[mp.Process], stop_queue: Optional[mp.Queue]) -> None:
    if process is None:
        return

    if stop_queue is not None and process.is_alive():
        try:
            stop_queue.put((_REQUEST_STOP,))
        except Exception:
            pass

    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)

    _close_queue(stop_queue)


def _start_child_process(
        ctx: mp.context.BaseContext,
        espeak_path: Optional[str],
) -> tuple[mp.Process, mp.Queue, mp.Queue]:
    child_input_queue = ctx.Queue()
    child_output_queue = ctx.Queue()
    child_process = ctx.Process(
        target=_phonemizer_child_main,
        args=(child_input_queue, child_output_queue, espeak_path),
        daemon=True,
        name="p2g-phonemizer-child",
    )
    child_process.start()
    return child_process, child_input_queue, child_output_queue


def _monitor_main(
        request_queue: mp.Queue,
        response_queue: mp.Queue,
        espeak_path: Optional[str],
        timeout_seconds: int,
) -> None:
    ctx = mp.get_context("spawn")
    child_process = None
    child_input_queue = None
    child_output_queue = None

    def restart_child() -> None:
        nonlocal child_process, child_input_queue, child_output_queue
        _stop_process(child_process, child_input_queue)
        _close_queue(child_output_queue)
        child_process, child_input_queue, child_output_queue = _start_child_process(ctx, espeak_path)

    def dispatch_to_child(request_type: str, request_id: int, payload: str | list[str], language: str):
        if child_process is None or not child_process.is_alive():
            response_queue.put((
                _STATUS_CHILD_CRASH,
                request_id,
                f"child exited before request started (exitcode={None if child_process is None else child_process.exitcode})",
            ))
            restart_child()
            return

        child_input_queue.put((request_type, request_id, payload, language))
        deadline = time.monotonic() + timeout_seconds

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                response_queue.put((_STATUS_TIMEOUT, request_id, f"child timed out after {timeout_seconds}s"))
                restart_child()
                return

            if not child_process.is_alive():
                response_queue.put((
                    _STATUS_CHILD_CRASH,
                    request_id,
                    f"child exited with code {child_process.exitcode}",
                ))
                restart_child()
                return

            try:
                child_response = child_output_queue.get(timeout=min(_POLL_INTERVAL_SECONDS, remaining))
            except queue.Empty:
                continue

            child_status, child_request_id, response_payload = child_response
            if child_request_id != request_id:
                response_queue.put((
                    _STATUS_MONITOR_ERROR,
                    request_id,
                    f"unexpected child response id {child_request_id} while waiting for {request_id}",
                ))
                restart_child()
                return

            response_queue.put((child_status, child_request_id, response_payload))
            return

    restart_child()

    try:
        while True:
            request = request_queue.get()
            request_type = request[0]
            if request_type == _REQUEST_STOP:
                return

            _, request_id, payload, language = request
            dispatch_to_child(request_type, request_id, payload, language)
    finally:
        _stop_process(child_process, child_input_queue)
        _close_queue(child_output_queue)


class PhonemizerSupervisor:
    def __init__(self, espeak_path: Optional[Path], timeout_seconds: int = 30):
        self._ctx = mp.get_context("spawn")
        self._espeak_path = None if espeak_path is None else str(espeak_path)
        self._timeout_seconds = timeout_seconds
        self._request_id = 0
        self._request_queue: Optional[mp.Queue] = None
        self._response_queue: Optional[mp.Queue] = None
        self._monitor_process: Optional[mp.Process] = None
        self.last_failure_kind: Optional[str] = None
        self.last_failure_detail: Optional[str] = None
        self.last_batch_failure_kind: Optional[str] = None
        self.last_batch_failure_detail: Optional[str] = None
        self.last_batch_used_fallback = False
        self.last_item_failure_counts: Counter[str] = Counter()
        self._start_monitor()

    def _start_monitor(self) -> None:
        self._request_queue = self._ctx.Queue()
        self._response_queue = self._ctx.Queue()
        self._monitor_process = self._ctx.Process(
            target=_monitor_main,
            args=(self._request_queue, self._response_queue, self._espeak_path, self._timeout_seconds),
            daemon=False,
            name="p2g-phonemizer-monitor",
        )
        self._monitor_process.start()

    def _restart_monitor(self) -> None:
        self.close()
        self._start_monitor()

    def _submit(self, request_type: str, payload: str | list[str], language: str) -> tuple[bool, str | list[str], str, str]:
        if self._request_queue is None or self._response_queue is None or self._monitor_process is None:
            raise RuntimeError("phonemizer supervisor has not been started")

        request_id = self._request_id
        self._request_id += 1
        self._request_queue.put((request_type, request_id, payload, language))
        deadline = time.monotonic() + self._timeout_seconds + _RESULT_GRACE_SECONDS

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = "monitor did not respond before the grace deadline"
                logger.warning("phonemizer monitor became unresponsive; restarting")
                self._restart_monitor()
                return False, payload if request_type == _REQUEST_BATCH else "", _STATUS_MONITOR_EXIT, detail

            if not self._monitor_process.is_alive():
                detail = f"monitor exited with code {self._monitor_process.exitcode}"
                logger.warning("phonemizer monitor exited unexpectedly; restarting")
                self._restart_monitor()
                return False, payload if request_type == _REQUEST_BATCH else "", _STATUS_MONITOR_EXIT, detail

            try:
                status, response_id, response_payload = self._response_queue.get(timeout=min(_POLL_INTERVAL_SECONDS, remaining))
            except queue.Empty:
                continue

            if response_id != request_id:
                raise RuntimeError(
                    f"received mismatched phonemizer response id {response_id}, expected {request_id}"
                )

            return status == _STATUS_OK, response_payload, status, "" if status == _STATUS_OK else str(response_payload)

    def transcribe(self, text: str, language: str) -> Optional[str]:
        self.last_failure_kind = None
        self.last_failure_detail = None
        success, payload, status, detail = self._submit(_REQUEST_SINGLE, text, language)
        if success:
            return payload  # type: ignore[return-value]
        self.last_failure_kind = status
        self.last_failure_detail = detail
        return None

    def transcribe_batch(self, texts: list[str], language: str) -> list[Optional[str]]:
        self.last_batch_failure_kind = None
        self.last_batch_failure_detail = None
        self.last_batch_used_fallback = False
        self.last_item_failure_counts = Counter()

        success, payload, status, detail = self._submit(_REQUEST_BATCH, texts, language)
        if success:
            return payload  # type: ignore[return-value]

        self.last_batch_failure_kind = status
        self.last_batch_failure_detail = detail
        self.last_batch_used_fallback = True

        outputs: list[Optional[str]] = []
        for text in texts:
            result = self.transcribe(text, language)
            outputs.append(result)
            if result is None:
                failure_kind = self.last_failure_kind or "unknown_failure"
                self.last_item_failure_counts[failure_kind] += 1
        return outputs

    def close(self) -> None:
        if self._monitor_process is None:
            return

        _stop_process(self._monitor_process, self._request_queue)
        _close_queue(self._response_queue)
        self._monitor_process = None
        self._request_queue = None
        self._response_queue = None
