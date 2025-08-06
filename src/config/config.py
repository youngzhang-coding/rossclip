import yaml
import threading

from src.constant import CONFIG_PATH

class ConfigSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    with open(CONFIG_PATH, 'r') as file:
                        cls._instance.config = yaml.safe_load(file)
        return cls._instance

    def get_config(self):
        return self.config

def get_config():
    return ConfigSingleton().get_config()