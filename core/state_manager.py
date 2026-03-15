from typing import Union, Dict
import pygame

from core import base_state, logger
from config import constants


class StateManager:
    """Manages the all states in the app.

    This class handles registering, setting and updating states.

    Attributes:
        states: Stores all the states.
        current_state: Keeping track of wich state is active.
    """

    def __init__(self):
        self.states: Dict[str, base_state.BaseState] = {}
        self.current_state: Union[base_state.BaseState, None] = None

    def register_state(self, state_name: str, state: base_state.BaseState):
        """Register a state and store it in memory.

        Args:
            state_name: The corresponding key to the state.
            state: The state object.

        Returns:
            None
        """
        if state_name in self.states or state in self.states.values():
            return

        self.states[state_name] = state

    def set_state(self, state_name: str, *args, **kwargs):
        """Activate the state.

        When a state is activated, the previous active state is deactivated.

        Args:
            state_name: The corresponding key to the state.

        Returns:
            None: Returns immediately if the state is not found or if it is already active,
                or if an internal error happens when transitioning states.
        """
        try:
            new_active_state = self.states.get(state_name)
            if new_active_state is None:
                return

            if new_active_state is self.current_state:
                return

            if self.current_state:
                self.current_state.exit()

            self.current_state = new_active_state
            self.current_state.enter(*args, **kwargs)
        except Exception:
            self.set_state(constants.Constants.state.error_state_string)
            logger.logger.exception(f"State switch failed")

    def update(self, dt: float, surface: pygame.Surface):
        """Update the active state.

        Args:
            dt: Time delta since last frame.
            surface: The surface to render to.

        Returns:
            None
        """
        if not self.current_state:
            return

        self.current_state.update(dt, surface)

    def handle_event(self, event: pygame.Event):
        """Passes the event to the current state to handle.

        Args:
            event: The event to handle.

        Returns:
            None
        """
        if not self.current_state:
            return

        self.current_state.handle_event(event)

    def close(self):
        """Exit the current state if there is one.

        Returns:
            None
        """
        if self.current_state is not None:
            self.current_state.exit()


state_manager = StateManager()
