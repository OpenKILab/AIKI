from petrel_client.client import Client
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock

class S3ClientManager:
    # TODO: redis
    _instance = None
    _lock = Lock()

    def __new__(cls, config_path='~/petreloss.conf', pool_size=8):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(S3ClientManager, cls).__new__(cls)
                cls._instance._initialize(config_path, pool_size)
        return cls._instance

    def _initialize(self, config_path, pool_size):
        self.pool = Queue(maxsize=pool_size)
        self.lock = Lock()
        for _ in range(pool_size):
            client = Client(config_path)
            self.pool.put(client)

    def get_client(self):
        with self.lock:
            return self.pool.get()

    def release_client(self, client):
        with self.lock:
            self.pool.put(client)

    def fetch_object(self, url):
        client = self.get_client()
        try:
            return client.get(url)
        finally:
            self.release_client(client)