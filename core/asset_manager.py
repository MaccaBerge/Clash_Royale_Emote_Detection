from typing import Tuple, Dict, Union, Optional
import pygame
from pygame.typing import ColorLike
import os
import pathlib

from core import logger
from config.constants import Constants


class AssetManager:
    """Loades, caches and retrives image and sound assets for the application.

    This class provides centralized asset handeling, including scanning asset
    directories, loading supported file types, image preproccesing and caching
    assets to avoid duplicates and improve performance. Missing assets are
    handled gracefully by returning fallback surfaces or warnings.
    """

    image_extentions = [
        ".bmp",
        ".gif",
        ".jpeg",
        ".lbm",
        ".pcx",
        ".png",
        ".pnm",
        ".svg",
        ".tga",
        ".tiff",
        ".webp",
        ".xpm",
    ]
    sound_extentions = [".wav", ".ogg", ".mp3"]

    def __init__(self):
        self._image_cache = {}
        self._sound_cache = {}

        root_project_path = pathlib.Path(__file__).parent.parent
        self._assets_folder_path = os.path.join(
            root_project_path, Constants.path.asset_folder_path
        )

    def load_folder(
        self,
        folder_path: Union[pathlib.Path, str],
        colorkey: Optional[ColorLike] = None,
    ) -> Tuple[Dict[str, pygame.Surface], Dict[str, pygame.mixer.Sound]]:
        """Loads every file with a supported file extension and saves them to cache.

        This method recursively loops thorugh every file in a directory. It checks
        for valid file extensions, distinguishes image from sound, loads the asset
        and saves it to the cache.

        Args:
            folder_path: Path to the directory to load, relative to the assset folder.
            colorkey: Color for setting the Pygame colorkey of an image.

        Returns:
            tuple: Two dictionaries containing the loaded image and sound files.
                - First dict maps image names (str) to `pygame.Surface` objects.
                - Second dict maps sound names (str) to `pygame.mixer.Sound` objects.

        """
        full_path = os.path.join(self._assets_folder_path, folder_path)
        if not os.path.exists(full_path):
            logger.logger.warning(f"Asset folder not found at: '{full_path}'")

            return {}, {}

        image_files = {}
        sound_files = {}

        for dir_path, _, file_names in os.walk(full_path):
            for file_name in file_names:
                name, file_extention = os.path.splitext(file_name)
                file_extention = file_extention.lower()

                if (
                    file_extention not in self.image_extentions
                    and file_extention not in self.sound_extentions
                ):
                    continue

                file_path = os.path.join(dir_path, file_name)

                if file_extention in self.image_extentions:
                    image_files[name] = self.load_image(
                        name, asset_path=file_path, colorkey=colorkey
                    )

                elif file_extention in self.sound_extentions:
                    sound_files[name] = self.load_sound(name, file_path)

        return image_files, sound_files

    def load_image(
        self,
        name: str,
        asset_path: Union[pathlib.Path, str],
        colorkey: Optional[ColorLike] = None,
    ) -> pygame.Surface:
        """Loads a single image and stores it in the cache.

        Loads an image from a file path relative to the asset folder and converts it
        to a `pygame.Surface` object. If the image already exists in the internal cache,
        the cached version is returned instead of reloading it.

        Args:
            name: The key associated with the image in the cache.
            asset_path: Path to the file to load, relative to the asset folder.
            colorkey: Color for setting the Pygame colorkey of an image. If set
                to None, no colorkey will be applied.

        Returns:
            A `pygame.Surface` object.

        Raises:
            FileNotFoundError: If the `asset_path` does not exist.
            pygame.error: If the image fails to load.
        """
        if name in self._image_cache:
            return self.get_image(name)

        full_path = os.path.join(self._assets_folder_path, asset_path)

        image = pygame.image.load(full_path).convert_alpha()

        if colorkey is not None:
            image.set_colorkey(colorkey)

        self.cache_image(name, image, name_exists_ok=True)
        return image

    def load_sound(
        self, name: str, asset_path: Union[pathlib.Path, str]
    ) -> Union[pygame.mixer.Sound, None]:
        """Loads a single sound and stores it in the cache.

        Loads a sound from a file path relative to the asset folder and converts
        it to a `pygame.mixer.Sound` object. If the sound already exists in the
        internal cache, the cached version is returned instead of reloading it.

        Args:
            name: The key associated with the sound in the cache.
            asset_path: Path to the file to load, relative to the asset folder.

        Returns:
            A `pygame.mixer.Sound` object.

        Raises:
            FileNotFoundError: If the `asset_path` does not exist.
            pygame.error: If the sound fails to load.
        """

        if name in self._sound_cache:
            return self.get_sound(name)

        full_path = os.path.join(self._assets_folder_path, asset_path)

        sound = pygame.mixer.Sound(full_path)

        self.cache_sound(name, sound, name_exists_ok=True)
        return sound

    def get_image(self, name: str) -> pygame.Surface:
        """Retrives an image from the internal cache.

        If the image exists in the internal cache, the cached image is returned.
        If the image does not exist in the internal cache, a placeholder image
        is returned.

        Args:
            name: The key associated with the sound in the cache.

        Returns:
            A `pygame.Surface` object.
        """
        if name in self._image_cache:
            return self._image_cache[name]
        else:
            logger.logger.warning(f"Name '{name}' not found in cache.")
            return pygame.Surface((100, 100))

    def get_sound(self, name):
        """Retrives a sound from the internal cache.

        Args:
            name: The key associated with the sound in the cache.

        Returns:
            If the sound exists in the internal cache, the method
            returns a `pygame.mixer.Sound` object. Otherwise None
            is returned.
        """
        return self._sound_cache.get(name)

    def cache_image(
        self, name: str, image: pygame.Surface, name_exists_ok: bool = False
    ):
        """Save image to the internal cache.

        Saves the image to the internal cache with `name` as its associated key.
        If `name_exists_ok` is set to True, a ValueError is raised if an image with
        the same `name` already exists in the internal cache.

        Args:
            name: The key associated with the image in the cache.
            image: The `pygame.Surface` to store in the cache.
            name_exists_ok: Decides if method raises a ValueError if an image with
                the same `name` already exists.

        Raises:
            ValueError: If name_exists_ok is set to False and `name` already exists
                as a key in the cache.
        """
        if name in self._image_cache and not name_exists_ok:
            raise ValueError(
                f"An image with the name '{name}' allready exists in the cache."
            )

        self._image_cache[name] = image

    def cache_sound(
        self, name: str, sound: pygame.mixer.Sound, name_exists_ok: bool = False
    ):
        """Save sound to the internal cache.

        Saves the sound to the internal cache with `name` as its associated key.
        If `name_exists_ok` is set to True, a ValueError is raised if a sound with
        the same `name` already exists in the internal cache.

        Args:
            name: The key associated with the sound in the cache.
            sound: The `pygame.mixer.Sound` to store in the cache.
            name_exists_ok: Decides if method raises a ValueError if a sound with
                the same `name` already exists.

        Raises:
            ValueError: If name_exists_ok is set to False and `name` already exists
                as a key in the cache.

        """
        if name in self._sound_cache and not name_exists_ok:
            raise ValueError(
                f"A sound with the name '{name}' allready exists in the cache."
            )

        self._sound_cache[name] = sound


asset_manager = AssetManager()
