import logging
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Optional


logger = logging.getLogger(__name__)

_REQUEST_JOB = "job"
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
        message_type = message[0]
        if message_type == _REQUEST_STOP:
            return

        _, request_id, text, language = message
        try:
            result = phonemize_romanize.strip_ipa(
                phonemize_romanize.phonemize_batch([text], language)[0]
            )
            output_queue.put((_STATUS_OK, request_id, result))
        except Exception as exc:
            output_queue.put((
                _STATUS_TRANSFORM_ERROR,
                request_id,
                f"{type(exc).__name__}: {exc}",
            ))


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

    if stop_queue is not None:
        try:
            stop_queue.close()
        except Exception:
            pass
        try:
            stop_queue.join_thread()
        except Exception:
            pass


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
        child_process, child_input_queue, child_output_queue = _start_child_process(ctx, espeak_path)

    restart_child()

    try:
        while True:
            request = request_queue.get()
            request_type = request[0]
            if request_type == _REQUEST_STOP:
                return

            _, request_id, text, language = request

            if child_process is None or not child_process.is_alive():
                response_queue.put((
                    _STATUS_CHILD_CRASH,
                    request_id,
                    f"child exited before request started (exitcode={None if child_process is None else child_process.exitcode})",
                ))
                restart_child()
                continue

            child_input_queue.put((_REQUEST_JOB, request_id, text, language))
            deadline = time.monotonic() + timeout_seconds

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    response_queue.put((
                        _STATUS_TIMEOUT,
                        request_id,
                        f"child timed out after {timeout_seconds}s",
                    ))
                    restart_child()
                    break

                if not child_process.is_alive():
                    response_queue.put((
                        _STATUS_CHILD_CRASH,
                        request_id,
                        f"child exited with code {child_process.exitcode}",
                    ))
                    restart_child()
                    break

                try:
                    child_response = child_output_queue.get(timeout=min(_POLL_INTERVAL_SECONDS, remaining))
                except queue.Empty:
                    continue

                child_status, child_request_id, payload = child_response
                if child_request_id != request_id:
                    response_queue.put((
                        _STATUS_MONITOR_ERROR,
                        request_id,
                        f"unexpected child response id {child_request_id} while waiting for {request_id}",
                    ))
                    restart_child()
                    break

                response_queue.put((child_status, child_request_id, payload))
                break
    finally:
        _stop_process(child_process, child_input_queue)


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

    def transcribe(self, text: str, language: str) -> Optional[str]:
        if self._request_queue is None or self._response_queue is None or self._monitor_process is None:
            raise RuntimeError("phonemizer supervisor has not been started")

        self.last_failure_kind = None
        self.last_failure_detail = None

        request_id = self._request_id
        self._request_id += 1
        self._request_queue.put((_REQUEST_JOB, request_id, text, language))
        deadline = time.monotonic() + self._timeout_seconds + _RESULT_GRACE_SECONDS

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.last_failure_kind = _STATUS_MONITOR_EXIT
                self.last_failure_detail = "monitor did not respond before the grace deadline"
                logger.warning("phonemizer monitor became unresponsive; restarting")
                self._restart_monitor()
                return None

            if not self._monitor_process.is_alive():
                self.last_failure_kind = _STATUS_MONITOR_EXIT
                self.last_failure_detail = f"monitor exited with code {self._monitor_process.exitcode}"
                logger.warning("phonemizer monitor exited unexpectedly; restarting")
                self._restart_monitor()
                return None

            try:
                status, response_id, payload = self._response_queue.get(
                    timeout=min(_POLL_INTERVAL_SECONDS, remaining)
                )
            except queue.Empty:
                continue

            if response_id != request_id:
                raise RuntimeError(
                    f"received mismatched phonemizer response id {response_id}, expected {request_id}"
                )

            if status == _STATUS_OK:
                return payload

            self.last_failure_kind = status
            self.last_failure_detail = payload
            return None

    def close(self) -> None:
        if self._monitor_process is None:
            return

        _stop_process(self._monitor_process, self._request_queue)
        if self._response_queue is not None:
            try:
                self._response_queue.close()
            except Exception:
                pass
            try:
                self._response_queue.join_thread()
            except Exception:
                pass
        self._monitor_process = None
        self._request_queue = None
        self._response_queue = None
