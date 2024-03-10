import pygame as pg
import numpy as np
from PyNAnts.config import Config
from PyNAnts.entities.food import Food


class PheromoneGrid:
    def __init__(self, screen_size):
        self.width = int(screen_size[0] / Config.PRATIO)
        self.height = int(screen_size[1] / Config.PRATIO)
        self.intensity = np.zeros((self.width, self.height, 3))

        self.image = pg.Surface((self.width, self.height)).convert()

    def update(self, dt):
        self.intensity.clip(max=1000)
        mask = self.intensity > 0
        self.intensity[mask] -= .2 * (60/Config.FPS) * ((dt/10) * Config.FPS)
        img_array = (self.intensity / 1000) * 255
        pg.surfarray.blit_array(self.image, img_array)
        return self.image

