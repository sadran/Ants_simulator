import pygame as pg


class Vec2:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def vint(self):
        return int(self.x), int(self.y)

    def __div__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x / other.x, self.y / other.y)
        if isinstance(other, int):
            return Vec2(self.x / other, self.y / other)
        else:
            raise TypeError("invalid type!")

    def __add__(self, other):
        if isinstance(other, Vec2):
            return Vec2(self.x + other.x, self.y + other.y)
        if isinstance(other, pg.math.Vector2):
            return Vec2(self.x + other[0], self.y + other[1])
        if isinstance(other, int):
            return Vec2(self.x + other, self.y + other)
        else:
            raise TypeError("invalid type!")

    def __idiv__(self, other):
        if isinstance(other, Vec2):
            self.x = int(self.x / other.x)
            self.y = int(self.y / other.y)
        if isinstance(other, int):
            self.x = int(self.x / other)
            self.y = int(self.y / other)
        else:
            raise TypeError("invalid type!")

    def __iadd__(self, other):
        if isinstance(other, Vec2):
            self.x += other.x
            self.y += other.x
        if isinstance(other, int):
            self.x += other
            self.y += other
        else:
            raise TypeError("invalid type!")
