import os
import queue
import threading
import numpy as np

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

_LIDAR_SENTINEL = None
_LIDAR_FLUSH_SIZE = 100000  # Większy bufor niż dla historii optymalizacji,
                          # bo rekordy są dużo mniejsze (7 liczb vs macierze)


class LidarHDF5Writer:
    """Asynchroniczny zapis trafień LiDAR-u do HDF5 (wzorzec producent-konsument).

    Producent (główny wątek symulacji) wywołuje ``put``, przekazując krotkę
    (time, drone_id, object_id, distance, hit_x, hit_y, hit_z).
    Konsument (wątek demon) buforuje rekordy i flushuje je do jednego
    datasetu rozszerzalnego w pliku ``lidar_hits.h5``.
    """

    # Nazwy kolumn zapisywane jako atrybut HDF5 dla dokumentacji
    COLUMNS = ["time", "drone_id", "object_id", "distance", "hit_x", "hit_y", "hit_z"]

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=2000)
        self._flush_counter = 0

        self._thread = threading.Thread(
            target=self._consumer_loop,
            name="LidarHDF5Writer",
            daemon=True,
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(self, record: tuple) -> None:
        """Wrzuca pojedynczy rekord do kolejki bez blokowania na długo."""
        try:
            self._queue.put(record, block=True, timeout=2.0)
        except queue.Full:
            # Upuszczamy dane zamiast blokować wątek symulacji
            pass

    def close(self) -> None:
        """Wysyła token kończący, czeka na drain kolejki i flush reszty."""
        self._queue.put(_LIDAR_SENTINEL)
        self._thread.join()

    # ------------------------------------------------------------------
    # Consumer internals
    # ------------------------------------------------------------------

    def _consumer_loop(self) -> None:
        buffer: list = []

        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is _LIDAR_SENTINEL:
                if buffer:
                    self._flush(buffer)
                break

            buffer.append(item)

            if len(buffer) >= _LIDAR_FLUSH_SIZE:
                self._flush(buffer)
                buffer.clear()

    def _flush(self, buffer: list) -> None:
        self._flush_counter += 1
        path = os.path.join(self.output_dir, "lidar_hits.h5")

        if _HAS_H5PY:
            self._flush_hdf5(buffer, path)
        else:
            self._flush_npz(buffer)

    def _flush_hdf5(self, buffer: list, path: str) -> None:
        # Konwersja listy krotek → macierz float32 (7 kolumn)
        # dtype float32 zamiast float64: ~2x mniej miejsca na dysku
        chunk = np.array(buffer, dtype=np.float32)  # shape: (N, 7)

        with h5py.File(path, "a") as f:
            if "hits" in f:
                ds = f["hits"]
                old_len = ds.shape[0]
                new_len = old_len + chunk.shape[0]
                ds.resize(new_len, axis=0)
                ds[old_len:new_len] = chunk
            else:
                ds = f.create_dataset(
                    "hits",
                    data=chunk,
                    maxshape=(None, 7),    # rozszerzalny wzdłuż osi wierszy
                    chunks=(4096, 7),      # chunk na ~28 KB przy float32
                    compression="gzip",
                    compression_opts=4,
                )
                # Dokumentacja kolumn jako atrybut HDF5
                ds.attrs["columns"] = self.COLUMNS

        print(f"[LIDAR] Flushed {len(buffer)} hits to HDF5 (chunk #{self._flush_counter}).")

    def _flush_npz(self, buffer: list) -> None:
        """Fallback gdy h5py niedostępne."""
        npz_dir = os.path.join(self.output_dir, "lidar_hits_npz")
        os.makedirs(npz_dir, exist_ok=True)
        chunk = np.array(buffer, dtype=np.float32)
        path = os.path.join(npz_dir, f"chunk_{self._flush_counter:04d}.npz")
        np.savez_compressed(path, hits=chunk, columns=self.COLUMNS)
        print(f"[LIDAR] Flushed {len(buffer)} hits to npz: {path}")