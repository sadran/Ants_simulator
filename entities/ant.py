import random
from math import pi, sin, cos, atan2, radians, degrees
import random
import pygame as pg
import numpy as np
from Ants_simulator.entities.vec2 import Vec2
from Ants_simulator.config import Config


class Ant(pg.sprite.Sprite):
    def __init__(self, drawSurf, pheroLayer):
        super().__init__()
        self.drawSurf = drawSurf
        self.curW, self.curH = self.drawSurf.get_size()
        self.pgSize = (int(self.curW/Config.PRATIO), int(self.curH/Config.PRATIO))
        self.isMyTrail = np.full(self.pgSize, False)
        self.phero = pheroLayer
        self.nest = Config.L_NEST
        self.image = pg.Surface((12, 21)).convert()
        self.image.set_colorkey(0)
        self.collected_food = 0
        self.delivared_food = 0
        cBrown = (100, 42, 42)
        # Draw Ant
        pg.draw.aaline(self.image, cBrown, [0, 5], [11, 15])
        pg.draw.aaline(self.image, cBrown, [0, 15], [11, 5]) # legs
        pg.draw.aaline(self.image, cBrown, [0, 10], [12, 10])
        pg.draw.aaline(self.image, cBrown, [2, 0], [4, 3]) # antena l
        pg.draw.aaline(self.image, cBrown, [9, 0], [7, 3]) # antena r
        pg.draw.ellipse(self.image, cBrown, [3, 2, 6, 6]) # head
        pg.draw.ellipse(self.image, cBrown, [4, 6, 4, 9]) # body
        pg.draw.ellipse(self.image, cBrown, [3, 13, 6, 8]) # rear
        # save drawing for later
        self.orig_img = pg.transform.rotate(self.image.copy(), -90)
        self.rect = self.image.get_rect(center=self.nest)
        self.ang = random.randint(0, 360)
        self.desireDir = pg.Vector2(cos(radians(self.ang)), sin(radians(self.ang)))
        self.pos = pg.Vector2(self.rect.center)
        self.vel = pg.Vector2(0, 0)
        self.mode = 0
        self.T = 0
        self.P = Config.P_MAX

    def update(self):
        pass

    def _avoid_edges(self):
        # Get locations to check as sensor points, in pairs for better detection.
        mid_sens = Vec2.vint(self.pos + pg.Vector2(21, 0).rotate(self.ang))
        left_sens = Vec2.vint(self.pos + pg.Vector2(15, -15).rotate(self.ang))
        right_sens = Vec2.vint(self.pos + pg.Vector2(15, 15).rotate(self.ang))
        # Avoid edges
        if not self.drawSurf.get_rect().collidepoint(left_sens) and self.drawSurf.get_rect().collidepoint(right_sens):
            self.desireDir = pg.Vector2(0, 1).rotate(self.ang)
        elif not self.drawSurf.get_rect().collidepoint(right_sens) and self.drawSurf.get_rect().collidepoint(left_sens):
            self.desireDir = pg.Vector2(0, -1).rotate(self.ang)
        elif not self.drawSurf.get_rect().collidepoint(mid_sens):
            self.desireDir = pg.Vector2(-1, 0).rotate(self.ang)

    @staticmethod
    def _get_sensing_vectors():
        lengths = [random.randint(1, Config.L_MAX) for _ in range(Config.X)]
        angles = [random.uniform(-Config.THETA_MAX, Config.THETA_MAX) for _ in range(Config.X)]
        vectors = [(l * np.cos(t), l * np.sin(t)) for l, t in zip(lengths, angles)]
        return vectors

    def _set_direction(self, dt):
        randAng = random.uniform(-Config.MU, Config.MU)
        randDir = pg.Vector2(cos(randAng), sin(randAng)).rotate(self.ang)
        self.desireDir = pg.Vector2(self.desireDir + randDir).normalize()
        self.vel = self.desireDir * Config.MAX_SPEED
        self.pos += self.vel * dt
        self.ang = degrees(atan2(self.vel[1], self.vel[0]))
        # adjusts angle of img to match heading
        self.image = pg.transform.rotate(self.orig_img, -self.ang)
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.center = self.pos

    def _sens_cell(self, pos):   # checks given points in Array, IDs, and pixels on screen.
        if self.drawSurf.get_rect().collidepoint(pos):
            sdpos = (int(pos[0]/Config.PRATIO), int(pos[1]/Config.PRATIO))
            array_r = self.phero.intensity[sdpos]
            ga_r = self.drawSurf.get_at(pos)[:3]
            isID = self.isMyTrail[sdpos]
        else:
            array_r = [0, 0, 0, 0]
            ga_r = [0, 0, 0]
            isID = 0
        return array_r, isID, ga_r

    def _sens_vectors(self, sensing_vectors):
        phero_intensity_list = []
        isID_list = []
        GA_result_list = []
        for vector in sensing_vectors:
            sensing_cell = Vec2.vint(self.pos + pg.Vector2(vector).rotate(self.ang))
            phero_intensity, isID, GA_result = self._sens_cell(sensing_cell)
            phero_intensity_list.append(phero_intensity)
            isID_list.append(isID)
            GA_result_list.append(GA_result)
        phero_intensity_list = np.array(phero_intensity_list)
        isID_list = np.array(isID_list)
        GA_result_list = np.array(GA_result_list)
        return phero_intensity_list, isID_list, GA_result_list


class WorkerAnt(Ant):
    def update(self, dt):  # behavior
        # Increase the simulation step since the last nest/food visit
        self.T += 1
        # Converts ant's current screen coordinates, to smaller resolution of pherogrid.
        scaledown_pos = (int(self.pos.x/Config.PRATIO), int(self.pos.y/Config.PRATIO))
        # Generate the sensing vectors
        sensing_vectors = self._get_sensing_vectors()
        # Sensing the environment
        phero_intensity_list, isID_list, GA_result_list = self._sens_vectors(sensing_vectors)

        # Don't search for food if you are not far enough from the nest
        if self.mode == 0 and self.pos.distance_to(self.nest) > 21:
            self.mode = 1
        # Look for food, or trail to food.
        elif self.mode == 1:
            # If the position is inside the surface
            if self.drawSurf.get_rect().collidepoint(self.pos):
                # If visited the nest, reset the simulation step
                if self.pos.distance_to(self.nest) < 24:
                    self.T = 0
                # If the existing home pheromone is less than yours, deposit home pheromone
                if self.phero.intensity[scaledown_pos][0] < Config.MAX_PHEROMONE_INTENSITY * np.exp(-Config.LAMBDA * self.T):
                    self.phero.intensity[scaledown_pos][0] = Config.MAX_PHEROMONE_INTENSITY * np.exp(-Config.LAMBDA * self.T)

                if Config.POLICY == "defending":
                    if self.phero.intensity[scaledown_pos][3] < Config.MAX_PHEROMONE_INTENSITY * np.exp(-Config.LAMBDA * self.P):
                        self.phero.intensity[scaledown_pos][3] = Config.MAX_PHEROMONE_INTENSITY * np.exp(-Config.LAMBDA * self.P)
                    if (phero_intensity_list[:, 1] > 0).any():
                        self.P = max(0, self.P - 1)
                    else:
                        self.P = min(Config.P_MAX, self.P + Config.P_MAX / Config.TP)
                self.isMyTrail[scaledown_pos] = True

            # Finding the most intense food pheromone among sensed cells
            self.find_best_direction(phero_intensity_list, sensing_vectors)

            # If food is found
            for GA in GA_result_list:
                if GA.tolist() == Config.FOOD_COLOR:
                    self.desireDir = pg.Vector2(-1, 0).rotate(self.ang).normalize()
                    self.collected_food += 1
                    self.mode = 2
                    self.T = 0
                    self.P = Config.P_MAX

        # Once found food, either follow own trail back to nest, or head in nest's general direction.
        elif self.mode == 2:
            # If the position is inside the surface
            if self.drawSurf.get_rect().collidepoint(self.pos):
                # If the existing home pheromone is less than yours, deposit home pheromone
                if self.phero.intensity[scaledown_pos][1] < Config.MAX_PHEROMONE_INTENSITY * np.exp(-Config.LAMBDA * self.T):
                    self.phero.intensity[scaledown_pos][1] = Config.MAX_PHEROMONE_INTENSITY * np.exp(-Config.LAMBDA * self.T)

            # Finding the most intense home pheromone among sensed cells
            home_phero_intensity_list = phero_intensity_list[:, 0]
            max_home_phero_intensity_arg = np.argmax(home_phero_intensity_list)
            # Find the corresponding sensing vector
            if home_phero_intensity_list[max_home_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_home_phero_intensity_arg]
                # Set direction
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

            if self.pos.distance_to(self.nest) < 24:
                self.desireDir = pg.Vector2(-1, 0).rotate(self.ang).normalize()
                self.isMyTrail[:] = False
                self.delivared_food += 1
                self.mode = 0
                self.T = 0

        # Avoid edges
        self._avoid_edges()
        self._set_direction(dt)

    def find_best_direction(self, phero_intensity_list, sensing_vectors):
        if Config.POLICY == "attacking":
            food_phero_intensity_list = phero_intensity_list[:, 1]
            max_food_phero_intensity_arg = np.argmax(food_phero_intensity_list)
            # Find the corresponding sensing vector
            if food_phero_intensity_list[max_food_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_food_phero_intensity_arg]
                # Set direction
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

        if Config.POLICY == "defending":
            food_phero_intensity_list = phero_intensity_list[:, 1]
            cout_phero_intensity_list = phero_intensity_list[:, 3]
            max_food_phero_intensity = 0
            max_food_phero_intensity_arg = 0
            for i, (food_phero, cout_phero) in enumerate(zip(food_phero_intensity_list, cout_phero_intensity_list)):
                if food_phero > max_food_phero_intensity and food_phero > cout_phero:
                    max_food_phero_intensity = food_phero
                    max_food_phero_intensity_arg = i

            # Find the corresponding sensing vector
            if food_phero_intensity_list[max_food_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_food_phero_intensity_arg]
                # Set direction
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)


class MaliciousAnt(Ant):
    def update(self, dt):  # behavior
        wandr_str = .12  # how random they walk around
        # Increase the simulation step since the last nest/food visit
        self.T += 1
        # Converts ant's current screen coordinates, to smaller resolution of pherogrid.
        scaledown_pos = (int(self.pos.x / Config.PRATIO), int(self.pos.y / Config.PRATIO))
        # Generate the sensing vectors
        sensing_vectors = self._get_sensing_vectors()
        # Sensing the environment
        phero_intensity_list, isID_list, GA_result_list = self._sens_vectors(sensing_vectors)

        # Don't search for food if you are not far enough from the nest
        if self.mode == 0 and self.pos.distance_to(self.nest) > 21:
            self.mode = 1
        # Look for food, or trail to food.
        elif self.mode == 1:
            # If the position is inside the surface
            if self.drawSurf.get_rect().collidepoint(self.pos):
                # If visited the nest, reset the simulation step
                if self.pos.distance_to(self.nest) < 24:
                    self.T = 0
                # If the existing food pheromone is less than yours, deposit home pheromone
                if self.phero.intensity[scaledown_pos][1] < Config.MAX_PHEROMONE_INTENSITY * np.exp(-Config.LAMBDA * self.T):
                    self.phero.intensity[scaledown_pos][1] = Config.MAX_PHEROMONE_INTENSITY * np.exp(-Config.LAMBDA * self.T)
                    self.phero.intensity[scaledown_pos][2] = Config.MAX_PHEROMONE_INTENSITY * np.exp(-Config.LAMBDA * self.T)
                self.isMyTrail[scaledown_pos] = True
            # Finding the most intense home pheromone among sensed cells
            self.find_best_direction(phero_intensity_list, sensing_vectors)

        # Avoid edges
        self._avoid_edges()
        self._set_direction(dt)

    def find_best_direction(self, phero_intensity_list, sensing_vectors):
        home_phero_intensity_list = phero_intensity_list[:, 0]
        max_home_phero_intensity_arg = np.argmax(home_phero_intensity_list)
        # Find the corresponding sensing vector
        if home_phero_intensity_list[max_home_phero_intensity_arg] != 0:
            best_vector = sensing_vectors[max_home_phero_intensity_arg]
            # Set direction
            self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

