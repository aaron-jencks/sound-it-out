from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import logging
import multiprocessing as mp
from pathlib import Path
import queue
import time
from typing import Optional


logger = logging.getLogger(__name__)

_REQUEST_BATCH = "batch"
_REQUEST_SINGLE = "single"
_REQUEST_STOP = "stop"
_STATUS_OK = "ok"
_POLL_INTERVAL_SECONDS = 0.05


@dataclass
class QueuedTransformBatch:
    language: str
    texts: list[str]


@dataclass
class CompletedTransformBatch:
    batch_id: int
    language: str
    texts: list[str]
    outputs: list[Optional[str]]
    event_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class _WorkerConfig:
    transform_type: str
    espeak_path: Optional[str] = None
    uroman_path: Optional[str] = None
    perl_path: Optional[str] = None


@dataclass
class _WorkerSlot:
    name: str
    process: mp.Process
    input_queue: mp.Queue
    output_queue: mp.Queue
    current_batch: Optional["_PendingBatch"] = None
    deadline: float = 0.0


@dataclass
class _PendingBatch:
    batch_id: int
    language: str
    texts: list[str]


def _transform_single(text: str, language: str, config: _WorkerConfig) -> str:
    from transcription.g2p import phonemize_romanize

    if config.transform_type == "phonemize":
        return phonemize_romanize.strip_ipa(
            phonemize_romanize.phonemize_batch([text], language)[0]
        )

    if config.transform_type == "romanize":
        if config.uroman_path is None or config.perl_path is None:
            raise ValueError("romanize worker requires uroman_path and perl_path")
        return phonemize_romanize.uromanize_batch(
            [text],
            config.uroman_path,
            config.perl_path,
        )[0]

    raise ValueError(f"unsupported transform type: {config.transform_type}")


def _transform_batch(texts: list[str], language: str, config: _WorkerConfig) -> list[str]:
    from transcription.g2p import phonemize_romanize

    if config.transform_type == "phonemize":
        return [
            phonemize_romanize.strip_ipa(item)
            for item in phonemize_romanize.phonemize_batch(texts, language)
        ]

    if config.transform_type == "romanize":
        if config.uroman_path is None or config.perl_path is None:
            raise ValueError("romanize worker requires uroman_path and perl_path")
        return phonemize_romanize.uromanize_batch(
            texts,
            config.uroman_path,
            config.perl_path,
        )

    raise ValueError(f"unsupported transform type: {config.transform_type}")


def _worker_main(
        input_queue: mp.Queue,
        output_queue: mp.Queue,
        config: _WorkerConfig,
) -> None:
    from transcription.g2p import phonemize_romanize

    if config.transform_type == "phonemize" and config.espeak_path is not None:
        phonemize_romanize.CustomEspeakBackend.set_library(config.espeak_path)
        phonemize_romanize.BACKENDS.clear()

    event_prefix = "phonemizer" if config.transform_type == "phonemize" else "romanize"

    while True:
        request = input_queue.get()
        request_type = request[0]
        if request_type == _REQUEST_STOP:
            return

        _, batch_id, language, payload = request
        if request_type == _REQUEST_SINGLE:
            try:
                output = _transform_single(payload, language, config)
                output_queue.put((_STATUS_OK, batch_id, [output], {}))
            except Exception:
                output_queue.put((
                    _STATUS_OK,
                    batch_id,
                    [None],
                    {f"{event_prefix}_transform_error": 1},
                ))
            continue

        if request_type != _REQUEST_BATCH:
            raise ValueError(f"unsupported request type: {request_type}")

        try:
            outputs = _transform_batch(payload, language, config)
            output_queue.put((_STATUS_OK, batch_id, outputs, {}))
        except Exception:
            event_counts = Counter({f"{event_prefix}_batch_fallback": 1})
            outputs: list[Optional[str]] = []
            for text in payload:
                try:
                    outputs.append(_transform_single(text, language, config))
                except Exception:
                    outputs.append(None)
                    event_counts[f"{event_prefix}_transform_error"] += 1
            output_queue.put((_STATUS_OK, batch_id, outputs, dict(event_counts)))


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
            stop_queue.put_nowait((_REQUEST_STOP,))
        except Exception:
            pass

    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)

    _close_queue(stop_queue)


class TransformPoolSupervisor:
    def __init__(
            self,
            transform_type: str,
            worker_count: int,
            timeout_seconds: int = 30,
            espeak_path: Optional[Path] = None,
            uroman_path: Optional[Path] = None,
            perl_path: Optional[str] = None,
    ):
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")

        self._ctx = mp.get_context("spawn")
        self._timeout_seconds = timeout_seconds
        self._worker_count = worker_count
        self._next_worker_index = 0
        self._batch_id = 0
        self._config = _WorkerConfig(
            transform_type=transform_type,
            espeak_path=None if espeak_path is None else str(espeak_path),
            uroman_path=None if uroman_path is None else str(uroman_path),
            perl_path=perl_path,
        )
        self._event_prefix = "phonemizer" if transform_type == "phonemize" else "romanize"
        self._workers: list[_WorkerSlot] = [self._start_worker(index) for index in range(worker_count)]

    def _start_worker(self, index: int) -> _WorkerSlot:
        input_queue = self._ctx.Queue(maxsize=1)
        output_queue = self._ctx.Queue(maxsize=1)
        process = self._ctx.Process(
            target=_worker_main,
            args=(input_queue, output_queue, self._config),
            daemon=False,
            name=f"p2g-transform-worker-{index}",
        )
        process.start()
        return _WorkerSlot(
            name=f"worker-{index}",
            process=process,
            input_queue=input_queue,
            output_queue=output_queue,
        )

    def _restart_worker(self, index: int) -> None:
        slot = self._workers[index]
        _stop_process(slot.process, slot.input_queue)
        _close_queue(slot.output_queue)
        replacement = self._start_worker(index)
        replacement.current_batch = None
        replacement.deadline = 0.0
        self._workers[index] = replacement

    def _build_completed_batch(
            self,
            batch: _PendingBatch,
            outputs: list[Optional[str]],
            event_counts: Counter[str],
    ) -> CompletedTransformBatch:
        return CompletedTransformBatch(
            batch_id=batch.batch_id,
            language=batch.language,
            texts=batch.texts,
            outputs=outputs,
            event_counts=event_counts,
        )

    def _run_single_recovery(
            self,
            worker_index: int,
            language: str,
            text: str,
    ) -> tuple[Optional[str], Counter[str]]:
        slot = self._workers[worker_index]
        if not slot.process.is_alive():
            self._restart_worker(worker_index)
            slot = self._workers[worker_index]

        request_id = -(self._batch_id + 1)
        self._batch_id += 1
        slot.input_queue.put((_REQUEST_SINGLE, request_id, language, text))
        deadline = time.monotonic() + self._timeout_seconds

        while True:
            if not slot.process.is_alive():
                self._restart_worker(worker_index)
                return None, Counter({f"{self._event_prefix}_child_crash": 1})

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._restart_worker(worker_index)
                return None, Counter({f"{self._event_prefix}_timeout": 1})

            try:
                status, response_id, outputs, event_counts = slot.output_queue.get(
                    timeout=min(_POLL_INTERVAL_SECONDS, remaining)
                )
            except queue.Empty:
                continue

            if status != _STATUS_OK or response_id != request_id:
                raise RuntimeError(
                    f"received mismatched worker recovery response {response_id}, expected {request_id}"
                )
            return outputs[0], Counter(event_counts)

    def _recover_failed_batch(
            self,
            worker_index: int,
            batch: _PendingBatch,
            failure_kind: str,
    ) -> CompletedTransformBatch:
        self._restart_worker(worker_index)
        event_counts = Counter({
            f"{self._event_prefix}_batch_fallback": 1,
            f"{self._event_prefix}_{failure_kind}": 1,
        })
        outputs: list[Optional[str]] = []
        for text in batch.texts:
            output, single_event_counts = self._run_single_recovery(worker_index, batch.language, text)
            outputs.append(output)
            event_counts.update(single_event_counts)
        return self._build_completed_batch(batch, outputs, event_counts)

    def _collect_one_result(self, worker_index: int) -> Optional[CompletedTransformBatch]:
        slot = self._workers[worker_index]
        batch = slot.current_batch
        if batch is None:
            if not slot.process.is_alive():
                self._restart_worker(worker_index)
            return None

        if not slot.process.is_alive():
            slot.current_batch = None
            slot.deadline = 0.0
            return self._recover_failed_batch(worker_index, batch, "child_crash")

        if time.monotonic() > slot.deadline:
            slot.current_batch = None
            slot.deadline = 0.0
            return self._recover_failed_batch(worker_index, batch, "timeout")

        try:
            status, response_id, outputs, event_counts = slot.output_queue.get_nowait()
        except queue.Empty:
            return None

        if status != _STATUS_OK or response_id != batch.batch_id:
            raise RuntimeError(
                f"received mismatched worker batch response {response_id}, expected {batch.batch_id}"
            )

        slot.current_batch = None
        slot.deadline = 0.0
        return self._build_completed_batch(batch, outputs, Counter(event_counts))

    def _drain_available(self) -> list[CompletedTransformBatch]:
        completed: list[CompletedTransformBatch] = []
        progress = True
        while progress:
            progress = False
            for worker_index in range(self._worker_count):
                result = self._collect_one_result(worker_index)
                if result is not None:
                    completed.append(result)
                    progress = True
        return completed

    def _try_enqueue(self, batch: _PendingBatch) -> bool:
        for offset in range(self._worker_count):
            worker_index = (self._next_worker_index + offset) % self._worker_count
            slot = self._workers[worker_index]
            if slot.current_batch is not None:
                continue
            if not slot.process.is_alive():
                self._restart_worker(worker_index)
                slot = self._workers[worker_index]

            try:
                slot.input_queue.put_nowait((_REQUEST_BATCH, batch.batch_id, batch.language, batch.texts))
            except queue.Full:
                continue

            slot.current_batch = batch
            slot.deadline = time.monotonic() + self._timeout_seconds
            self._next_worker_index = (worker_index + 1) % self._worker_count
            return True
        return False

    def pump(self, batch: Optional[QueuedTransformBatch] = None) -> list[CompletedTransformBatch]:
        completed = self._drain_available()
        if batch is None:
            return completed

        pending_batch = _PendingBatch(
            batch_id=self._batch_id,
            language=batch.language,
            texts=batch.texts,
        )
        self._batch_id += 1

        while not self._try_enqueue(pending_batch):
            completed.extend(self.wait_for_completion())
        return completed

    def wait_for_completion(self) -> list[CompletedTransformBatch]:
        while True:
            completed = self._drain_available()
            if completed:
                return completed
            time.sleep(_POLL_INTERVAL_SECONDS)

    def has_inflight_batches(self) -> bool:
        return any(slot.current_batch is not None for slot in self._workers)

    def close(self) -> None:
        for slot in self._workers:
            _stop_process(slot.process, slot.input_queue)
            _close_queue(slot.output_queue)
        self._workers = []
