import pygame as pg
import numpy as np


class PheromoneGrid:
    def __init__(self, screen_size, config):
        self.config = config
        self.width = int(screen_size[0] / self.config.PRATIO)
        self.height = int(screen_size[1] / self.config.PRATIO)
        self.intensity = np.zeros((self.width, self.height, 4))

        self.image = pg.Surface((self.width, self.height)).convert()

    def update(self, dt):
        self.intensity.clip(max=self.config.MAX_PHEROMONE_INTENSITY)
        non_zero_mask = self.intensity >= (dt * self.config.K)
        zero_mask = self.intensity < (dt * self.config.K)

        self.intensity[non_zero_mask] -= (dt * self.config.K)
        self.intensity[zero_mask] = 0

        img_array = self.__calculate_image_array()
        pg.surfarray.blit_array(self.image, img_array)
        return self.image

    def __calculate_image_array(self):
        home_pheromone_color = np.full((self.width, self.height, 3), self.config.HOME_PHEROMONE_COLOR)
        food_pheromone_color = np.full((self.width, self.height, 3), self.config.FOOD_PHEROMONE_COLOR)
        mis_pheromone_color = np.full((self.width, self.height, 3), self.config.MIS_PHEROMONE_COLOR)
        cout_pheromone_color = np.full((self.width, self.height, 3), self.config.CAUT_PHEROMONE_COLOR)
        pheromone_grid = np.zeros((self.width, self.height, 3))

        if self.config.SHOW_HOME_PHERO:
            pheromone_grid += home_pheromone_color[:, :, :] * np.broadcast_to(self.intensity[:, :, 0, np.newaxis], (
                self.width, self.height, 3)) / self.config.MAX_PHEROMONE_INTENSITY

        if self.config.SHOW_FOOD_PHERO:
            pheromone_grid += food_pheromone_color[:, :, :] * np.broadcast_to(self.intensity[:, :, 1, np.newaxis], (
                self.width, self.height, 3)) / self.config.MAX_PHEROMONE_INTENSITY

        if self.config.SHOW_MIS_PHERO:
            pheromone_grid += mis_pheromone_color[:, :, :] * np.broadcast_to(self.intensity[:, :, 2, np.newaxis], (
                self.width, self.height, 3)) / self.config.MAX_PHEROMONE_INTENSITY

        if self.config.SHOW_CAUT_PHERO:
            pheromone_grid += cout_pheromone_color[:, :, :] * np.broadcast_to(self.intensity[:, :, 3, np.newaxis], (
                self.width, self.height, 3)) / self.config.MAX_PHEROMONE_INTENSITY

        return pheromone_grid
