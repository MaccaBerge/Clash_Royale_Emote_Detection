import pygame
import cv2
import threading

from core import base_state, state_manager
from config.constants import Constants
from components.text import Text


class StartupState(base_state.BaseState):
    def __init__(self, app):
        self.app = app
        self.video_input = None
        self.state_text = Text(
            "",
            position=(
                pygame.display.get_window_size()[0] / 2,
                pygame.display.get_window_size()[1] / 2,
            ),
            font_path=Constants.path.main_font_path,
        )
        self.succes = None
        self.error = None
        self.init_done = False

    def _error(self, error):
        self.error = str(error)
        self.state_text.text = self.error

    def _initialize_video_input(self):
        self.state_text.text = "Fetching Video Input..."
        self.error = None

        def video_init():
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                self.video_input = cap
                self.succes = True
                return
            self._error("Error initializing video input.")

        threading.Thread(target=video_init, daemon=True).start()

    def enter(self):
        if self.init_done:
            return

        self._initialize_video_input()

        self.init_done = True

    def update(self, dt: float, surface: pygame.Surface):
        if self.succes:
            self.app.set_video_input(self.video_input)
            state_manager.state_manager.set_state(
                Constants.state.detection_state_string
            )
            return
        self.render(surface)

    def render(self, surface: pygame.Surface):
        surface.fill((255, 255, 255))
        self.state_text.render(surface)

    def handle_event(self, event: pygame.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and self.error:
                self._initialize_video_input()
