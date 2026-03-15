from pygame import Surface, Event
from abc import abstractmethod
from typing import Union


class BaseState:
    """Abstract base class for application states.

    All states must subclass this class and implement the core lifecycle
    methods: `enter`, `exit`, `update`, `handle_event`.

    Lifecycle:
        * `enter()`: Called when the state becomes active.
        * `exit()`: Called when the state is no longer active.
        * `update(dt, screen)`: Called once per frame to update the state logic and render to screen.
        * `handle_event(event)`: Called for each Pygame event.

    Child classes can choose wich methods to override.
    """

    @abstractmethod
    def enter(self, *args, **kwargs):
        """Called when the state becomes active."""
        pass

    @abstractmethod
    def exit(self):
        """Called when the state is deactivated"""
        pass

    @abstractmethod
    def update(self, dt: float, surface: Surface) -> Union[str, None]:
        """Called once per frame to update the state logic.

        Args:
            dt: Time delta since last frame, in seconds.

        """
        pass

    @abstractmethod
    def handle_event(self, event: Event):
        """Handles events from the Pygame event loop.

        Args:
            event: A Pygame event to handle.
        """
        pass
