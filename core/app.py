import pygame
import cv2
from sys import exit

pygame.init()
pygame.mixer.init()

from config import settings, constants
from core import state_manager, asset_manager, logger
from states import startup, detection, error_screen
from components.timer import timer_manager


class App:
    """Main application class responsible for initializing Pygame,
    managing states, load assets and run the main update loop.

    Attributes:
        screen (pygame.Surface): Main display surface.
        clock (pygame.time.Clock): Main frame timer handler.
        video_input (:obj: `cv2.VideoCapture`, optional): Object to capture frames. Defaults to None.
        running (bool): Controls the main loop. When set to True the main loop runs.
    """

    def __init__(self):
        pygame.mixer.set_num_channels(
            constants.Constants.general.pygame_mixer_num_channels
        )

        self.screen = pygame.display.set_mode(settings.settings.screen.size, vsync=True)
        pygame.display.set_caption(constants.Constants.general.program_name)
        self.clock = pygame.time.Clock()

        self.video_input = None

        self.running = True

    def _setup_states(self):
        """Register the the different states of the program."""
        state_manager.state_manager.register_state(
            constants.Constants.state.startup_state_string, startup.StartupState(self)
        )
        state_manager.state_manager.register_state(
            constants.Constants.state.detection_state_string,
            detection.DetectionState(self),
        )
        state_manager.state_manager.set_state(
            constants.Constants.state.startup_state_string
        )

    def _load_assets(self):
        """Loades assets, resize them and store them in cache."""
        images, _ = asset_manager.asset_manager.load_folder("")

        for name, image in images.items():
            image_width = image.width
            image_height = image.height

            small_image_ratio = (
                settings.settings.general.small_image_width / image_width
            )
            big_image_ratio = settings.settings.screen.size[1] / image_height

            small_image = pygame.transform.scale_by(
                image, small_image_ratio
            ).convert_alpha()
            big_image = pygame.transform.scale_by(
                image, big_image_ratio
            ).convert_alpha()

            small_image_name = f"{name}_small"
            big_image_name = f"{name}_big"

            asset_manager.asset_manager.cache_image(small_image_name, small_image)
            asset_manager.asset_manager.cache_image(big_image_name, big_image)

    def set_video_input(self, video_input: cv2.VideoCapture):
        if self.video_input is None:
            self.video_input = video_input

    def get_video_input(self):
        return self.video_input

    def startup(self):
        """Organizes loading assets, setting up states and running the main loop."""
        self._load_assets()
        self._setup_states()

        try:
            self.run()
        except Exception:
            error_state = error_screen.ErrorScreen()
            error_state.enter(
                error_message="Hmm... seams like i didnt care enough to expect this error during coding time. Well, well... not my problem, have fun restarting!"
            )

    def shutdown(self):
        if state_manager.state_manager is not None:
            state_manager.state_manager.close()

        pygame.quit()
        exit()

    def run(self):
        """Run the main program loop

        Updates the global timer_manager and state_manager.
        """
        while self.running:
            dt = self.clock.tick(settings.settings.screen.fps) / 1000
            timer_manager.update()
            state_manager.state_manager.update(dt, self.screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.shutdown()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.shutdown()

                state_manager.state_manager.handle_event(event)

            pygame.display.update()
