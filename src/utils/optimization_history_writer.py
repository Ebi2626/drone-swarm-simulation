import os
import queue
import threading
import time

import numpy as np

_SENTINEL = None
_BUFFER_FLUSH_SIZE = 100

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False


class OptimizationHistoryWriter:
    """Asynchronous writer for optimization history data (Producer-Consumer pattern).

    The producer (main thread) calls ``put_generation_data`` to enqueue NumPy
    matrices.  A background daemon thread consumes the queue, accumulates
    entries in a local buffer, and flushes them to disk every
    ``_BUFFER_FLUSH_SIZE`` items (HDF5 when *h5py* is available, compressed
    ``.npz`` otherwise).
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._flush_counter = 0

        self._thread = threading.Thread(
            target=self._consumer_loop,
            name="OptimizationHistoryWriter",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put_generation_data(self, data: dict) -> None:
        """Enqueue a generation snapshot without blocking the caller longer
        than necessary.

        Parameters
        ----------
        data : dict
            Must contain NumPy arrays, e.g.
            ``{"objectives_matrix": np.ndarray, "decisions_matrix": np.ndarray}``.
        """
        self._queue.put(data, block=True, timeout=5.0)

    def close(self) -> None:
        """Send a poison-pill, wait for the consumer to drain, and flush
        any remaining buffered data."""
        self._queue.put(_SENTINEL)
        self._thread.join()

    # ------------------------------------------------------------------
    # Consumer internals
    # ------------------------------------------------------------------

    def _consumer_loop(self) -> None:
        buffer: list[dict] = []

        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _SENTINEL:
                # Flush leftover data before exiting
                if buffer:
                    self._flush(buffer)
                break

            buffer.append(item)

            if len(buffer) >= _BUFFER_FLUSH_SIZE:
                self._flush(buffer)
                buffer.clear()

    def _flush(self, buffer: list[dict]) -> None:
        self._flush_counter += 1
        timestamp = int(time.time() * 1000)
        chunk_name = f"chunk_{self._flush_counter:04d}_{timestamp}"

        if _HAS_H5PY:
            self._flush_hdf5(buffer, chunk_name)
        else:
            self._flush_npz(buffer, chunk_name)

    # ---- HDF5 backend ------------------------------------------------

    def _flush_hdf5(self, buffer: list[dict], chunk_name: str) -> None:
        path = os.path.join(self.output_dir, "optimization_history.h5")
        keys = buffer[0].keys()

        with h5py.File(path, "a") as f:
            for key in keys:
                # ZMIANA: Używamy np.stack zamiast np.concatenate, aby zachować 
                # wymiar generacji (Generacja x Osobnik x Cechy)
                stacked = np.stack([entry[key] for entry in buffer], axis=0)

                if key in f:
                    ds = f[key]
                    old_len = ds.shape[0]
                    new_len = old_len + stacked.shape[0]
                    ds.resize(new_len, axis=0)
                    ds[old_len:new_len] = stacked
                else:
                    maxshape = (None,) + stacked.shape[1:]
                    f.create_dataset(
                        key,
                        data=stacked,
                        maxshape=maxshape,
                        chunks=True,
                        compression="gzip",
                        compression_opts=4,
                    )
        print(f"[HISTORY] Flushed {len(buffer)} generations to HDF5.")

    # ---- NPZ backend ------------------------------------------------

    def _flush_npz(self, buffer: list[dict], chunk_name: str) -> None:
        npz_dir = os.path.join(self.output_dir, "optimization_history_npz")
        os.makedirs(npz_dir, exist_ok=True)

        keys = buffer[0].keys()
        # ZMIANA: Używamy np.stack również w fallbacku
        arrays = {
            key: np.stack([entry[key] for entry in buffer], axis=0)
            for key in keys
        }

        path = os.path.join(npz_dir, f"{chunk_name}.npz")
        np.savez_compressed(path, **arrays)
        print(f"[HISTORY] Flushed {len(buffer)} generations to npz: {path}")