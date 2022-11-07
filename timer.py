import time

class Timer():
    def __init__(self):
        self.start_time = time.time()

    def start(self):
        self.start_time = time.time()

    def elapsed_time(self):
        current_time = time.time()

        duration = current_time - self.start_time
    
        hours = int(duration / 3600)
        minutes = int((duration % 3600) / 60)
        seconds = int((duration % 3600) % 60)

        return f"{hours}h {minutes}m {seconds}s"