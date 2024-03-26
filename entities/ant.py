import random
from math import pi, sin, cos, atan2, radians, degrees
import random
import pygame as pg
import numpy as np
from Ants_simulator.entities.vec2 import Vec2


class Ant(pg.sprite.Sprite):
    def __init__(self, drawSurf, pheroLayer, config):
        super().__init__()
        self.config = config
        self.drawSurf = drawSurf
        self.curW, self.curH = self.drawSurf.get_size()
        self.pgSize = (int(self.curW / self.config.PRATIO), int(self.curH / self.config.PRATIO))
        self.isMyTrail = np.full(self.pgSize, False)
        self.phero = pheroLayer
        self.nest = self.config.L_NEST
        self.image = pg.Surface((12, 21)).convert()
        self.image.set_colorkey(0)
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
        self.P = self.config.P_MAX

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

    def _get_sensing_vectors(self):
        lengths = [random.randint(1, self.config.L_MAX) for _ in range(self.config.X)]
        angles = [random.uniform(-self.config.THETA_MAX, self.config.THETA_MAX) for _ in range(self.config.X)]
        vectors = [(l * np.cos(t), l * np.sin(t)) for l, t in zip(lengths, angles)]
        return vectors

    def _set_direction(self, dt):
        rand_ang = random.uniform(-self.config.MU, self.config.MU)
        rand_dir = pg.Vector2(cos(rand_ang), sin(rand_ang)).rotate(self.ang)
        self.desireDir = pg.Vector2(self.desireDir + rand_dir).normalize()
        self.vel = self.desireDir * self.config.MAX_SPEED
        self.pos += self.vel * dt
        self.ang = degrees(atan2(self.vel[1], self.vel[0]))
        # adjusts angle of img to match heading
        self.image = pg.transform.rotate(self.orig_img, -self.ang)
        self.rect = self.image.get_rect(center=self.rect.center)
        self.rect.center = self.pos

    def _sens_cell(self, pos):   # checks given points in Array, IDs, and pixels on screen.
        if self.drawSurf.get_rect().collidepoint(pos):
            sdpos = (int(pos[0] / self.config.PRATIO), int(pos[1] / self.config.PRATIO))
            array_r = self.phero.intensity[sdpos]
            ga_r = self.drawSurf.get_at(pos)[:3]
            is_id = self.isMyTrail[sdpos]
        else:
            array_r = [0, 0, 0, 0]
            ga_r = [0, 0, 0]
            is_id = 0
        return array_r, is_id, ga_r

    def _sens_vectors(self, sensing_vectors):
        phero_intensity_list = []
        is_id_list = []
        ga_result_list = []
        for vector in sensing_vectors:
            sensing_cell = Vec2.vint(self.pos + pg.Vector2(vector).rotate(self.ang))
            phero_intensity, is_id, ga_result = self._sens_cell(sensing_cell)
            phero_intensity_list.append(phero_intensity)
            is_id_list.append(is_id)
            ga_result_list.append(ga_result)
        phero_intensity_list = np.array(phero_intensity_list)
        is_id_list = np.array(is_id_list)
        ga_result_list = np.array(ga_result_list)
        return phero_intensity_list, is_id_list, ga_result_list


class WorkerAnt(Ant):
    def __init__(self, drawSurf, pheroLayer, config):
        super().__init__(drawSurf, pheroLayer, config)
        self.collected_food = 0
        self.delivered_food = 0

    def update(self, dt):  # behavior
        # Increase the simulation step since the last nest/food visit
        self.T += 1
        # Converts ant's current screen coordinates, to smaller resolution of pherogrid.
        scaledown_pos = (int(self.pos.x / self.config.PRATIO), int(self.pos.y / self.config.PRATIO))
        # Generate the sensing vectors
        sensing_vectors = self._get_sensing_vectors()
        # Sensing the environment
        phero_intensity_list, is_id_list, ga_result_list = self._sens_vectors(sensing_vectors)

        if self.config.POLICY == 0:
            self._policy_0(scaledown_pos, phero_intensity_list, sensing_vectors, ga_result_list, dt)
        elif self.config.POLICY == 1:
            self._policy_1(scaledown_pos, phero_intensity_list, sensing_vectors, ga_result_list, dt)
        elif self.config.POLICY == 2:
            self._policy_2(scaledown_pos, phero_intensity_list, sensing_vectors, ga_result_list, dt)

    def _policy_0(self, pos, phero_intensity_list, sensing_vectors, ga_result_list, dt):
        # Don't search for food if you are not far enough from the nest
        if self.mode == 0 and self.pos.distance_to(self.nest) > 21:
            self.mode = 1

        # Look for food, or trail to food.
        elif self.mode == 1:
            if self.drawSurf.get_rect().collidepoint(self.pos):
                # If visited the nest, reset the simulation step
                if self.pos.distance_to(self.nest) < 24:
                    self.T = 0
                # Secrete home pheromone
                if self.phero.intensity[pos][0] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T):
                    self.phero.intensity[pos][0] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)
                self.isMyTrail[pos] = True

            food_phero_intensity_list = phero_intensity_list[:, 1]
            max_food_phero_intensity_arg = np.argmax(food_phero_intensity_list)
            # Find the corresponding sensing vector
            if food_phero_intensity_list[max_food_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_food_phero_intensity_arg]
                # Set direction
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

            # If food is found
            for ga in ga_result_list:
                if ga.tolist() == self.config.FOOD_COLOR:
                    self.desireDir = pg.Vector2(-1, 0).rotate(self.ang).normalize()
                    self.collected_food += 1
                    self.mode = 2
                    self.T = 0
                    self.P = self.config.P_MAX

        elif self.mode == 2:
            # If the position is inside the surface
            if self.drawSurf.get_rect().collidepoint(self.pos):
                # If the existing home pheromone is less than yours, deposit home pheromone
                if self.phero.intensity[pos][1] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T):
                    self.phero.intensity[pos][1] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)

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
                self.delivered_food += 1
                self.mode = 0
                self.T = 0

        # Avoid edges
        self._avoid_edges()
        self._set_direction(dt)

    def _policy_1(self, pos, phero_intensity_list, sensing_vectors, ga_result_list, dt):
        # Don't search for food if you are not far enough from the nest
        if self.mode == 0 and self.pos.distance_to(self.nest) > 21:
            self.mode = 1

        if self.mode == 1:
            # If the position is inside the surface
            if self.drawSurf.get_rect().collidepoint(self.pos):
                # If visited the nest, reset the simulation step
                if self.pos.distance_to(self.nest) < 24:
                    self.T = 0
                # Secrete home pheromone
                if self.phero.intensity[pos][0] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T):
                    self.phero.intensity[pos][0] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)
                # Secrete cautionary pheromone
                if self.phero.intensity[pos][3] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.P):
                    self.phero.intensity[pos][3] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.P)
                if (phero_intensity_list[:, 1] > 0).any():
                    self.P = max(0, self.P - 1)
                else:
                    self.P = min(self.config.P_MAX, self.P + self.config.P_MAX / self.config.TP)
                self.isMyTrail[pos] = True

            food_phero_intensity_list = phero_intensity_list[:, 1]
            caut_phero_intensity_list = phero_intensity_list[:, 3]
            max_food_phero_intensity = 0
            max_food_phero_intensity_arg = 0
            for i, (food_phero, caut_phero) in enumerate(zip(food_phero_intensity_list, caut_phero_intensity_list)):
                if food_phero > max_food_phero_intensity and food_phero > caut_phero:
                    max_food_phero_intensity = food_phero
                    max_food_phero_intensity_arg = i

            # Find the corresponding sensing vector
            if food_phero_intensity_list[max_food_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_food_phero_intensity_arg]
                # Set direction
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

            # If food is found
            for ga in ga_result_list:
                if ga.tolist() == self.config.FOOD_COLOR:
                    self.desireDir = pg.Vector2(-1, 0).rotate(self.ang).normalize()
                    self.collected_food += 1
                    self.mode = 2
                    self.T = 0
                    self.P = self.config.P_MAX

        if self.mode == 2:
            # If the position is inside the surface
            if self.drawSurf.get_rect().collidepoint(self.pos):
                # If the existing home pheromone is less than yours, deposit home pheromone
                if self.phero.intensity[pos][1] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T):
                    self.phero.intensity[pos][1] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)

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
                self.delivered_food += 1
                self.mode = 0
                self.T = 0

        # Avoid edges
        self._avoid_edges()
        self._set_direction(dt)

    def _policy_2(self, pos, phero_intensity_list, sensing_vectors, ga_result_list, dt):
        # Don't search for food if you are not far enough from the nest
        if self.mode == 0 and self.pos.distance_to(self.nest) > 21:
            self.mode = 1

        if self.mode == 1:
            # If the position is inside the surface
            if self.drawSurf.get_rect().collidepoint(self.pos):
                # If visited the nest, reset the simulation step
                if self.pos.distance_to(self.nest) < 24:
                    self.T = 0
                # Secrete home pheromone
                if self.phero.intensity[pos][0] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T):
                    self.phero.intensity[pos][0] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)
                # Secrete cautionary pheromone
                c = np.exp(-self.config.LAMBDA * self.T)
                if self.phero.intensity[pos][3] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.P) * c:
                    self.phero.intensity[pos][3] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.P) * c
                if (phero_intensity_list[:, 1] > 0).any():
                    self.P = max(0, self.P - 1)
                else:
                    self.P = min(self.config.P_MAX, self.P + self.config.P_MAX / self.config.TP)
                self.isMyTrail[pos] = True

            food_phero_intensity_list = phero_intensity_list[:, 1]
            caut_phero_intensity_list = phero_intensity_list[:, 3]
            max_food_phero_intensity = 0
            max_food_phero_intensity_arg = 0
            for i, (food_phero, caut_phero) in enumerate(zip(food_phero_intensity_list, caut_phero_intensity_list)):
                if food_phero > max_food_phero_intensity and food_phero > 1.1 * caut_phero:
                    max_food_phero_intensity = food_phero
                    max_food_phero_intensity_arg = i

            # Find the corresponding sensing vector
            if food_phero_intensity_list[max_food_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_food_phero_intensity_arg]
                # Set direction
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

            # If food is found
            for ga in ga_result_list:
                if ga.tolist() == self.config.FOOD_COLOR:
                    self.desireDir = pg.Vector2(-1, 0).rotate(self.ang).normalize()
                    self.collected_food += 1
                    self.mode = 2
                    self.T = 0
                    self.P = self.config.P_MAX

        if self.mode == 2:
            # If the position is inside the surface
            if self.drawSurf.get_rect().collidepoint(self.pos):
                # If the existing home pheromone is less than yours, deposit home pheromone
                if self.phero.intensity[pos][1] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T):
                    self.phero.intensity[pos][1] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)

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
                self.delivered_food += 1
                self.mode = 0
                self.T = 0

        # Avoid edges
        self._avoid_edges()
        self._set_direction(dt)


class MaliciousAnt(Ant):
    def update(self, dt):  # behavior
        wandr_str = .12  # how random they walk around
        # Increase the simulation step since the last nest/food visit
        self.T += 1
        # Converts ant's current screen coordinates, to smaller resolution of pherogrid.
        scaledown_pos = (int(self.pos.x / self.config.PRATIO), int(self.pos.y / self.config.PRATIO))
        # Generate the sensing vectors
        sensing_vectors = self._get_sensing_vectors()
        # Sensing the environment
        phero_intensity_list, is_id_list, ga_result_list = self._sens_vectors(sensing_vectors)

        if self.config.POLICY == 0:
            self._policy_0(scaledown_pos, phero_intensity_list, sensing_vectors, dt)

        elif self.config.POLICY == 1:
            self._policy_1(scaledown_pos, phero_intensity_list, sensing_vectors, dt)
        elif self.config.POLICY == 2:
            self._policy_2(scaledown_pos, phero_intensity_list, sensing_vectors, dt)

    def _policy_0(self, pos, phero_intensity_list, sensing_vectors, dt):
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
                if self.phero.intensity[pos][1] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T):
                    self.phero.intensity[pos][1] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)
                    self.phero.intensity[pos][2] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)
                self.isMyTrail[pos] = True
            # Finding the most intense home pheromone among sensed cells
            home_phero_intensity_list = phero_intensity_list[:, 0]
            max_home_phero_intensity_arg = np.argmax(home_phero_intensity_list)
            # Find the corresponding sensing vector
            if home_phero_intensity_list[max_home_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_home_phero_intensity_arg]
                # Set direction
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

        # Avoid edges
        self._avoid_edges()
        self._set_direction(dt)

    def _policy_1(self, pos, phero_intensity_list, sensing_vectors, dt):
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
                if self.phero.intensity[pos][1] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T):
                    self.phero.intensity[pos][1] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)
                    self.phero.intensity[pos][2] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)
                self.isMyTrail[pos] = True
            # Finding the most intense home pheromone among sensed cells
            home_phero_intensity_list = phero_intensity_list[:, 0]
            max_home_phero_intensity_arg = np.argmax(home_phero_intensity_list)
            # Find the corresponding sensing vector
            if home_phero_intensity_list[max_home_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_home_phero_intensity_arg]
                # Set direction
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

        # Avoid edges
        self._avoid_edges()
        self._set_direction(dt)

    def _policy_2(self, pos, phero_intensity_list, sensing_vectors, dt):
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
                if self.phero.intensity[pos][1] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T):
                    self.phero.intensity[pos][1] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)
                    self.phero.intensity[pos][2] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)
                self.isMyTrail[pos] = True
            # Finding the most intense home pheromone among sensed cells
            home_phero_intensity_list = phero_intensity_list[:, 0]
            max_home_phero_intensity_arg = np.argmax(home_phero_intensity_list)
            # Find the corresponding sensing vector
            if home_phero_intensity_list[max_home_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_home_phero_intensity_arg]
                # Set direction
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

        # Avoid edges
        self._avoid_edges()
        self._set_direction(dt)

    def _policy_3(self, pos, phero_intensity_list, ga_result_list, sensing_vectors, dt):
        # Don't search for food if you are not far enough from the nest
        if self.mode == 0 and self.pos.distance_to(self.nest) > 21:
            self.mode = 1

        elif self.mode == 1:
            # If the position is inside the surface
            if self.drawSurf.get_rect().collidepoint(self.pos):
                for ga in ga_result_list:
                    if ga.tolist() == self.config.FOOD_COLOR:
                        self.mode = 2
                # Finding the most intense home pheromone among sensed cells
                food_phero_intensity_list = phero_intensity_list[:, 1]
                max_food_phero_intensity_arg = np.argmax(food_phero_intensity_list)
                # Find the corresponding sensing vector
                if food_phero_intensity_list[max_food_phero_intensity_arg] != 0:
                    best_vector = sensing_vectors[max_food_phero_intensity_arg]
                    # Set direction
                    self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

        elif self.mode == 2:
            # If the position is inside the surface
            if self.drawSurf.get_rect().collidepoint(self.pos):
                if self.phero.intensity[pos][0] < self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T):
                    self.phero.intensity[pos][0] = self.config.MAX_PHEROMONE_INTENSITY * np.exp(
                        -self.config.LAMBDA * self.T)

                for ga in ga_result_list:
                    if ga.tolist() == self.config.FOOD_COLOR:
                        self.T = 0
                self.isMyTrail[pos] = True
            # Finding the most intense home pheromone among sensed cells
            home_phero_intensity_list = phero_intensity_list[:, 0]
            max_home_phero_intensity_arg = np.argmax(home_phero_intensity_list)
            # Find the corresponding sensing vector
            if home_phero_intensity_list[max_home_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_home_phero_intensity_arg]
                # Set direction
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)

        # Avoid edges
        self._avoid_edges()
        self._set_direction(dt)
