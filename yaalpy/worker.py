import threading
import multiprocessing
from queue import Queue, Empty
import traceback
from typing import Generic, TypeVar, Callable

T = TypeVar('T')
class SyncVar(Generic[T]):
    def __init__(self, value: T):
        self._value: T = value
        self._lock: threading.Lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._value
    
    def set(self, value: T):
        with self._lock:
            self._value = value

class Worker:
    _total_workers: int = 0

    def __init__(self, maxsize: int = 0):
        Worker._total_workers += 1
        self._jobs: Queue = Queue(maxsize)
        self._alive: SyncVar[bool] = SyncVar[bool](True)
        self._execute_remaining_jobs: SyncVar[bool] = SyncVar[bool](True)

        self._thread: threading.Thread = threading.Thread(
            target=self, name=f"Worker {self._total_workers}", daemon=True
        )
        self._thread.start()

    def __call__(self):
        while self._execute_remaining_jobs.get() and (
            not self._jobs.empty() or self._alive.get()
        ):
            try:
                job = self._jobs.get(block=True, timeout=1)
                try:
                    job()
                except BaseException as e:
                    print("")
                    traceback.print_exception(e)
                self._jobs.task_done()
            except Empty:
                pass

    def print_cpu(self):
        def job():
            process = multiprocessing.current_process()
            print(f"Worker is using thread \"{self._thread.name}\" is using cpu {process.name}")
        
        self.add_job(job)
    
    def add_job(self, job: Callable[[], None]) -> bool:
        if self._alive.get():
            self._jobs.put(job)
            return True
        
        return False

    def join_thread(self, forced: bool = False):
        """
        Prohibits the addition of new jobs and wait until the thread is finished.\n
        If forced is True, the worker doesn't executes the remaining jobs (but finishes the current job it is executing).
        """
        self._alive.set(False)
        
        if forced:
            self._execute_remaining_jobs.set(False)

        self._thread.join()
    
    def join_queue(self):
        self._jobs.join()

    def queue_size(self) -> int:
        """
        Return the approximate size of the queue (not reliable!).
        """
        return self._jobs.qsize()

if __name__ == "__main__":
    worker = Worker()

    worker.print_cpu()

    def attendre(nom, duree):
        def job():
            import time
            time.sleep(duree)
            print(nom, "fini")
        
        worker.add_job(job)

    for i in range(4):
        attendre(f"worker {i}", 1)
    
    print("queue join...")
    worker.join_queue()
    print("queue joined")

    attendre(f"worker 4", 1)

    print("queue join again...")
    worker.join_queue()
    print("queue joined")

    print("thread join...")
    worker.join_thread()
    print("thread joined")
