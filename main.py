import pygame
from sys import exit

from core.app import App
from core.logger import logger

if __name__ == "__main__":
    try:
        App().startup()
    except Exception as e:
        logger.critical(
            "Application failed to start due to a fatel error", exc_info=True
        )
        pygame.quit() 
        exit()
