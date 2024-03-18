from math import pi, sin, cos
import pygame as pg
from Ants_simulator.config import Config
from entities.pheromoneGrid import PheromoneGrid
from entities.ant import WorkerAnt, MaliciousAnt
from entities.food import Food


def main():
    pg.init()  # prepare window
    pg.display.set_caption("NAnts")

    # setup full screen or window mode
    if Config.FLLSCRN:
        current_rez = (pg.display.Info().current_w, pg.display.Info().current_h)
        screen = pg.display.set_mode(current_rez, pg.SCALED | pg.NOFRAME | pg.FULLSCREEN, vsync=Config.VSYNC)
    else:
        screen = pg.display.set_mode((Config.WIDTH, Config.HEIGHT), pg.SCALED, vsync=Config.VSYNC)

    cur_w, cur_h = screen.get_size()
    screen_size = (cur_w, cur_h)

    colony = pg.sprite.Group()
    world = PheromoneGrid(screen_size)

    malicious_ants_num = 0
    worker_ants_num = 0
    for n in range(Config.ANTS):
        if n < Config.ANTS * Config.MAL_ANT_FRC:
            colony.add(MaliciousAnt(screen, world))
            malicious_ants_num += 1
        else:
            colony.add(WorkerAnt(screen, world))
            worker_ants_num += 1
    clock = pg.time.Clock()

    """foodBits = 200
    cx, cy = Config.L_FOOD
    fRadius = Config.R_FOOD
    for i in range(0, foodBits):  # spawn food bits evenly within a circle
        dist = pow(i / (foodBits - 1.0), 0.5) * fRadius
        angle = 2 * pi * 0.618033 * i
        fx = cx + dist * cos(angle)
        fy = cy + dist * sin(angle)
        foods.add(Food((fx, fy)))"""

    # main loop
    sim_step = 0
    while sim_step < Config.N:
        sim_step += 1
        for e in pg.event.get():
            if e.type == pg.QUIT or e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                return

        dt = clock.tick(Config.FPS)/1000
        pheroImg = world.update(dt)
        colony.update(dt)
        total_collected_food = sum(ant.collected_food for ant in colony.sprites() if isinstance(ant, WorkerAnt))
        total_delivered_food = sum(ant.delivared_food for ant in colony.sprites() if isinstance(ant, WorkerAnt))
        ants_collected_ever = 0
        ants_delivered_ever = 0
        for ant in colony.sprites():
            if isinstance(ant, WorkerAnt) and ant.delivared_food != 0:
                ants_delivered_ever += 1
            if isinstance(ant, WorkerAnt) and ant.collected_food != 0:
                ants_collected_ever += 1

        ants_collected_ever_ratio = ants_collected_ever / worker_ants_num
        ants_delivered_ever_ratio = ants_delivered_ever / worker_ants_num

        screen.fill(0)  # fill MUST be after sensors update, so previous draw is visible to them
        rescaled_img = pg.transform.scale(pheroImg, screen_size)
        pg.Surface.blit(screen, rescaled_img, (0, 0))
        pg.draw.circle(screen, Config.HOME_COLOR, Config.L_NEST, Config.R_NEST / Config.PRATIO)
        pg.draw.circle(screen, Config.FOOD_COLOR, Config.L_FOOD, Config.R_FOOD / Config.PRATIO)
        font = pg.font.Font(None, 30)
        screen.blit(font.render(f"simulation step: {sim_step}", True, [0, 200, 0]), (8, 8))
        screen.blit(font.render(f"total collected foods:{total_collected_food}", True, [0, 200, 0]), (8, 30))
        screen.blit(font.render(f"total delivered foods:{total_delivered_food}", True, [0, 200, 0]), (8, 60))
        screen.blit(font.render(f"ants ever collected ratio:{ants_collected_ever_ratio:.2f}", True, [0, 200, 0]), (8, 90))
        screen.blit(font.render(f"ants ever delivered ratio:{ants_delivered_ever_ratio:.2f}", True, [0, 200, 0]), (8, 120))

        #colony.draw(screen)
        pg.display.update()


if __name__ == '__main__':
    main()
    pg.quit()
