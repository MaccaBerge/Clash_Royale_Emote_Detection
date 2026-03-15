import pygame
import traceback
from sys import exit

from components import text
from config import settings, constants
from core import logger


class ErrorScreen:
    def __init__(self):
        self.screen = None
        self.error_message = constants.Constants.error_screen.default_error_message
        self.error_message_text = text.Text(
            constants.Constants.error_screen.default_error_message,
            (
                settings.settings.screen.size[0] / 2,
                20,
            ),
            constants.Constants.path.main_font_path,
            color=constants.Constants.error_screen.text_color,
            anchor="n",
            wrap_length=int(settings.settings.screen.size[0] * 0.97),
        )

    def _shutdown(self):
        pygame.quit()
        exit()

    def enter(
        self,
        error_message: str = constants.Constants.error_screen.default_error_message,
        include_traceback: bool = True,
    ):
        self.screen = pygame.display.get_surface()
        message = error_message

        if include_traceback:
            traceback_text = traceback.format_exc()
            if traceback_text and message.strip() != "NoneType: None":
                message += "\n\n" + traceback_text

        self.error_message_text.text = message

        self.run()

    def run(self):
        if self.screen is None:
            logger.logger.warning("Pygame display not found for error screen")
            return
        while True:
            self.screen.fill(constants.Constants.error_screen.background_color)

            self.error_message_text.render(self.screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._shutdown()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self._shutdown()

            pygame.display.update()
