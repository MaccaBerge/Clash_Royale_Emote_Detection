from typing import Literal, Iterable, Union
import pygame


class Text:
    def __init__(
        self,
        text: str,
        position: Iterable[Union[int, float]],
        font_path: str,
        font_size=20,
        color=(0, 0, 0),
        anchor: Literal["center", "n", "s", "e", "w"] = "center",
        wrap_length=0,
    ):
        self._text = text
        self.position = tuple(position)
        self.font = self._load_font(font_path, font_size)
        self.color = color
        self.anchor = anchor
        self.wrap_length = wrap_length

        self._rendered_text_surface = None
        self._rect = None

        self._render_text()

    def _load_font(self, font_path, font_size):
        return pygame.font.Font(font_path, font_size)

    def _render_text(self):
        self._rendered_text_surface = self.font.render(
            self._text, True, self.color, wraplength=self.wrap_length
        )

        anchor_map = {
            "center": "center",
            "n": "midtop",
            "s": "midbottom",
            "e": "midright",
            "w": "midleft",
        }

        if self.anchor not in anchor_map:
            raise ValueError(
                f"Invalid anchor '{self.anchor}'. Expected: 'center', 'n', 's', 'e', 'w'."
            )

        rect_arg = {anchor_map[self.anchor]: self.position}
        self._rect = self._rendered_text_surface.get_rect(**rect_arg)

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        value = str(value)

        if value != self._text:
            self._text = value
            self._render_text()

    def render(self, surface: pygame.Surface):
        if self._rendered_text_surface is None or self._rect is None:
            return

        surface.blit(self._rendered_text_surface, self._rect)
