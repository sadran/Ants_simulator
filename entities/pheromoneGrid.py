import pygame as pg
import numpy as np
from Ants_simulator.config import Config
from Ants_simulator.entities.food import Food


class PheromoneGrid:
    def __init__(self, screen_size):
        self.width = int(screen_size[0] / Config.PRATIO)
        self.height = int(screen_size[1] / Config.PRATIO)
        self.intensity = np.zeros((self.width, self.height, 3))
        self.T = np.zeros((self.width, self.height, 3))

        self.image = pg.Surface((self.width, self.height)).convert()

    def update(self, dt):
        self.intensity.clip(max=Config.MAX_PHEROMONE_INTENSITY)
        non_zero_mask = self.intensity >= dt
        zero_mask = self.intensity < dt

        self.intensity[non_zero_mask] -= dt
        self.intensity[zero_mask] = 0

        #img_array = self.__calculate_image_array()
        img_array = self.intensity * 255 / Config.MAX_PHEROMONE_INTENSITY
        pg.surfarray.blit_array(self.image, img_array)
        return self.image

    def __calculate_image_array(self):
        home_pheromone_color = np.full((self.width, self.height, 3), Config.HOME_PHEROMONE_COLOR)
        food_pheromone_color = np.full((self.width, self.height, 3), Config.FOOD_PHEROMONE_COLOR)
        mis_pheromone_color = np.full((self.width, self.height, 3), Config.MIS_PHEROMONE_COLOR)
        pheromone_grid = np.zeros((self.width, self.height, 3))

        pheromone_grid[:, :, 0] += self.intensity[:, :, 0] * home_pheromone_color[:, :, 0]
        pheromone_grid[:, :, 1] += self.intensity[:, :, 0] * home_pheromone_color[:, :, 1]
        pheromone_grid[:, :, 2] += self.intensity[:, :, 0] * home_pheromone_color[:, :, 2]

        pheromone_grid[:, :, 0] += self.intensity[:, :, 1] * food_pheromone_color[:, :, 0]
        pheromone_grid[:, :, 1] += self.intensity[:, :, 1] * food_pheromone_color[:, :, 1]
        pheromone_grid[:, :, 2] += self.intensity[:, :, 1] * food_pheromone_color[:, :, 2]

        pheromone_grid[:, :, 0] += self.intensity[:, :, 2] * mis_pheromone_color[:, :, 0]
        pheromone_grid[:, :, 1] += self.intensity[:, :, 2] * mis_pheromone_color[:, :, 1]
        pheromone_grid[:, :, 2] += self.intensity[:, :, 2] * mis_pheromone_color[:, :, 2]
        img_array = (pheromone_grid / Config.MAX_PHEROMONE_INTENSITY) * 255
        return img_array