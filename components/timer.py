from typing import Union, List, Callable
import time


class Timer:
    def __init__(self, duration: Union[int, float], callback, repeating: bool = True):
        self.duration = duration
        self.callback = callback
        self.repeating = repeating
        self.finished = False

        self.next_trigger_time = None

    def start(self):
        self.finished = False
        current_time = time.time()
        self.next_trigger_time = current_time + self.duration

    def stop(self):
        self.finished = True
        self.next_trigger_time = None

    def reset(self):
        self.start()

    def update(self):
        if self.finished:
            return
        if self.next_trigger_time is None:
            return

        current_time = time.time()

        if current_time > self.next_trigger_time:
            self.callback()
            if self.repeating:
                self.next_trigger_time += self.duration
            else:
                self.finished = True


class TimerManager:
    def __init__(self):
        self.timers: List[Timer] = []

    def create_timer(
        self, duration: Union[int, float], callback: Callable, repeating: bool = True
    ):
        timer = Timer(duration=duration, callback=callback, repeating=repeating)
        timer.start()
        self.timers.append(timer)
        return timer

    def update(self):
        for timer in self.timers:
            timer.update()

        self.timers = [timer for timer in self.timers if not timer.finished]


timer_manager = TimerManager()
