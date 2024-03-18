import pygame as pg
import numpy as np
from Ants_simulator.config import Config
from Ants_simulator.entities.food import Food


class PheromoneGrid:
    def __init__(self, screen_size):
        self.width = int(screen_size[0] / Config.PRATIO)
        self.height = int(screen_size[1] / Config.PRATIO)
        self.intensity = np.zeros((self.width, self.height, 4))

        self.image = pg.Surface((self.width, self.height)).convert()

    def update(self, dt):
        self.intensity.clip(max=Config.MAX_PHEROMONE_INTENSITY)
        non_zero_mask = self.intensity >= (dt * Config.K)
        zero_mask = self.intensity < (dt * Config.K)

        self.intensity[non_zero_mask] -= (dt * Config.K)
        self.intensity[zero_mask] = 0

        img_array = self.__calculate_image_array()
        pg.surfarray.blit_array(self.image, img_array)
        return self.image

    def __calculate_image_array(self):
        home_pheromone_color = np.full((self.width, self.height, 3), Config.HOME_PHEROMONE_COLOR)
        food_pheromone_color = np.full((self.width, self.height, 3), Config.FOOD_PHEROMONE_COLOR)
        mis_pheromone_color = np.full((self.width, self.height, 3), Config.MIS_PHEROMONE_COLOR)
        cout_pheromone_color = np.full((self.width, self.height, 3), Config.COUT_PHEROMONE_COLOR)
        pheromone_grid = np.zeros((self.width, self.height, 3))

        if Config.POLICY == "attacking":
            pheromone_grid += home_pheromone_color[:, :, :] * np.broadcast_to(self.intensity[:, :, 0, np.newaxis], (
                self.width, self.height, 3)) / Config.MAX_PHEROMONE_INTENSITY

        pheromone_grid += food_pheromone_color[:, :, :] * np.broadcast_to(self.intensity[:, :, 1, np.newaxis], (
            self.width, self.height, 3)) / Config.MAX_PHEROMONE_INTENSITY

        pheromone_grid += mis_pheromone_color[:, :, :] * np.broadcast_to(self.intensity[:, :, 2, np.newaxis], (
            self.width, self.height, 3)) / Config.MAX_PHEROMONE_INTENSITY

        if Config.POLICY == "defending":
            pheromone_grid += cout_pheromone_color[:, :, :] * np.broadcast_to(self.intensity[:, :, 3, np.newaxis], (
                self.width, self.height, 3)) / Config.MAX_PHEROMONE_INTENSITY

        return pheromone_grid
