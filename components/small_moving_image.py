from typing import Union, List, Tuple, Iterable
import pygame
from math import sqrt


class SmallMovingImage:
    def __init__(
        self,
        image: pygame.Surface,
        position: Iterable[Union[int, float]],
        speed: Union[int, float],
        direction: Iterable[Union[int, float]] = [0, 1],
    ):
        self.image = image
        self.rect = self.image.get_rect(center=tuple(position))
        self.speed = speed
        self.direction = direction

    def update(self, dt: float):
        self.move(dt, speed=self.speed, direction=self.direction)

    def render(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)

    def move(
        self,
        dt,
        speed: Union[int, float] = 10,
        direction: Iterable[Union[int, float]] = [0, 1],
    ):
        direction = tuple(direction)
        dir_length = sqrt(direction[0] ** 2 + direction[1] ** 2)
        dir_norm = [direction[0] / dir_length, direction[1] / dir_length]
        dx = dir_norm[0] * speed * dt
        dy = dir_norm[1] * speed * dt

        self.rect.centerx += dx
        self.rect.centery += dy


class SmallMovingImageManager:
    def __init__(self):
        self.moving_images: List[SmallMovingImage] = []

    def create_image(
        self,
        image: pygame.Surface,
        position: Iterable[Union[int, float]],
        speed: Union[int, float],
        direction: Iterable[Union[int, float]] = [0, 1],
    ):
        moving_image = SmallMovingImage(image, position, speed, direction=direction)
        self.moving_images.append(moving_image)

    def update(self, dt: float, surface: pygame.Surface):
        for moving_image in self.moving_images:
            moving_image.update(dt)
            moving_image.render(surface)

        self.moving_images = [
            moving_image
            for moving_image in self.moving_images
            if moving_image.rect.top < surface.height
        ]
