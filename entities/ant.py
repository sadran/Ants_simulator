import random
from math import pi, sin, cos, atan2, radians, degrees
from random import randint
import pygame as pg
import numpy as np
from PyNAnts.entities.vec2 import Vec2
from PyNAnts.config import Config


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
        self.ang = randint(0, 360)
        self.desireDir = pg.Vector2(cos(radians(self.ang)), sin(radians(self.ang)))
        self.pos = pg.Vector2(self.rect.center)
        self.vel = pg.Vector2(0, 0)
        self.last_sdp = (self.nest[0]/10/2, self.nest[1]/10/2)
        self.mode = 0

    def update(self):
        pass

    def _avoid_edges(self, max_speed, wandr_str, steer_str):
        # Get locations to check as sensor points, in pairs for better detection.
        mid_sens = Vec2.vint(self.pos + pg.Vector2(21, 0).rotate(self.ang))
        left_sens = Vec2.vint(self.pos + pg.Vector2(18, -8).rotate(self.ang))
        right_sens = Vec2.vint(self.pos + pg.Vector2(18, 8).rotate(self.ang))
        # Avoid edges
        if not self.drawSurf.get_rect().collidepoint(left_sens) and self.drawSurf.get_rect().collidepoint(right_sens):
            self.desireDir += pg.Vector2(0, 1).rotate(self.ang)  # .normalize()
            wandr_str = .01
            steer_str = 5
        elif not self.drawSurf.get_rect().collidepoint(right_sens) and self.drawSurf.get_rect().collidepoint(left_sens):
            self.desireDir += pg.Vector2(0, -1).rotate(self.ang)  # .normalize()
            wandr_str = .01
            steer_str = 5
        elif not self.drawSurf.get_rect().collidepoint(mid_sens):
            self.desireDir += pg.Vector2(-1, 0).rotate(self.ang)  # .normalize()
            max_speed = 5
            wandr_str = .01
            steer_str = 5
        return max_speed, wandr_str, steer_str

    @staticmethod
    def _get_sensing_vectors():
        lengths = [random.randint(1, Config.L_MAX) for _ in range(Config.X)]
        angles = [random.uniform(-Config.THETA_MAX, Config.THETA_MAX) for _ in range(Config.X)]
        vectors = [(l * np.cos(t), l * np.sin(t)) for l, t in zip(lengths, angles)]
        return vectors

    def _set_direction(self, dt, max_speed, wandr_str, steer_str):
        randAng = randint(0, 360)
        randDir = pg.Vector2(cos(radians(randAng)),sin(radians(randAng)))
        self.desireDir = pg.Vector2(self.desireDir + randDir * wandr_str).normalize()
        dzVel = self.desireDir * max_speed
        dzStrFrc = (dzVel - self.vel) * steer_str
        accel = dzStrFrc if pg.Vector2(dzStrFrc).magnitude() <= steer_str else pg.Vector2(dzStrFrc.normalize() * steer_str)
        velo = self.vel + accel * dt
        self.vel = velo if pg.Vector2(velo).magnitude() <= max_speed else pg.Vector2(velo.normalize() * max_speed)
        self.pos += self.vel * dt
        self.ang = degrees(atan2(self.vel[1], self.vel[0]))
        # adjusts angle of img to match heading
        self.image = pg.transform.rotate(self.orig_img, -self.ang)
        self.rect = self.image.get_rect(center=self.rect.center)  # recentering fix
        # actually update position
        self.rect.center = self.pos

    def _sens_cell(self, pos):   # checks given points in Array, IDs, and pixels on screen.
        if self.drawSurf.get_rect().collidepoint(pos):
            sdpos = (int(pos[0]/Config.PRATIO), int(pos[1]/Config.PRATIO))
            array_r = self.phero.intensity[sdpos]
            ga_r = self.drawSurf.get_at(pos)[:3]
            isID = self.isMyTrail[sdpos]
        else:
            array_r = [0, 0, 0]
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
        rand_str = 0.12  # how random they walk around
        steer_str = 3  # 3 or 4, dono
        max_speed = 12
        # Converts ant's current screen coordinates, to smaller resolution of pherogrid.
        scaledown_pos = (int(self.pos.x/Config.PRATIO), int(self.pos.y/Config.PRATIO))

        sensing_vectors = self._get_sensing_vectors()

        phero_intensity_list, isID_list, GA_result_list = self._sens_vectors(sensing_vectors)

        if self.mode == 0 and self.pos.distance_to(self.nest) > 21:
            self.mode = 1

        elif self.mode == 1:  # Look for food, or trail to food.
            if self.drawSurf.get_rect().collidepoint(self.pos):
                self.phero.intensity[scaledown_pos][0] += 250
                self.isMyTrail[scaledown_pos] = True
                self.last_sdp = scaledown_pos

            food_phero_intensity_list = phero_intensity_list[:, 1]
            max_food_phero_intensity_arg = np.argmax(food_phero_intensity_list)
            if food_phero_intensity_list[max_food_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_food_phero_intensity_arg]
                self.desireDir = pg.Vector2(best_vector).normalize().rotate(self.ang)
                rand_str = .1

            for GA in GA_result_list:
                if GA.tolist() == Config.FOOD_COLOR:
                    self.desireDir = pg.Vector2(-1, 0).rotate(self.ang).normalize()
                    max_speed = 5
                    rand_str = .01
                    steer_str = 5
                    self.mode = 2

        elif self.mode == 2:    # Once found food, either follow own trail back to nest, or head in nest's general direction.
            if scaledown_pos[0] in range(0, self.pgSize[0]) and scaledown_pos[1] in range(0, self.pgSize[1]):
                self.phero.intensity[scaledown_pos][1] += 250
                self.last_sdp = scaledown_pos

            if self.pos.distance_to(self.nest) < 24:
                self.desireDir = pg.Vector2(-1, 0).rotate(self.ang).normalize()
                self.isMyTrail[:] = False
                max_speed = 5
                rand_str = .01
                steer_str = 5
                self.mode = 0
            else:
                self.desireDir += pg.Vector2(self.nest - self.pos).normalize() * .08
                rand_str = .1
        # Avoid edges
        max_speed, rand_str, steer_str = self._avoid_edges(max_speed, rand_str, steer_str)
        self._set_direction(dt, max_speed, rand_str, steer_str)


class MaliciousAnt(Ant):
    def update(self, dt):  # behavior
        wandr_str = .12  # how random they walk around
        max_speed = 12  # 10-12 seems ok
        steer_str = 3  # 3 or 4, dono
        # Converts ant's current screen coordinates, to smaller resolution of pherogrid.
        scaledown_pos = (int(self.pos.x / Config.PRATIO), int(self.pos.y / Config.PRATIO))

        sensing_vectors = self._get_sensing_vectors()

        phero_intensity_list, isID_list, GA_result_list = self._sens_vectors(sensing_vectors)

        if self.mode == 0 and self.pos.distance_to(self.nest) > 21:
            self.mode = 1

        elif self.mode == 1:  # Look for food, or trail to food.
            if scaledown_pos != self.last_sdp and self.drawSurf.get_rect().collidepoint(self.pos):
                self.phero.intensity[scaledown_pos][1] += 250
                self.phero.intensity[scaledown_pos][2] += 250
                self.isMyTrail[scaledown_pos] = True
                self.last_sdp = scaledown_pos

            food_phero_intensity_list = phero_intensity_list[:, 1]
            max_food_phero_intensity_arg = np.argmax(food_phero_intensity_list)
            if food_phero_intensity_list[max_food_phero_intensity_arg] != 0:
                best_vector = sensing_vectors[max_food_phero_intensity_arg]
                self.desireDir += pg.Vector2(best_vector).rotate(self.ang).normalize()
                wandr_str = .1

        # Avoid edges
        max_speed, wandr_str, steer_str = self._avoid_edges(max_speed, wandr_str, steer_str)
        self._set_direction(dt, max_speed, wandr_str, steer_str)
