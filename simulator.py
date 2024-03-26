import pygame as pg
from entities.pheromoneGrid import PheromoneGrid
from entities.ant import WorkerAnt, MaliciousAnt


def simulator(config, result_directory=None):
    pg.init()  # prepare window
    pg.display.set_caption("ants")
    if result_directory is not None:
        image_directory = result_directory.joinpath("screan_images")
        image_directory.mkdir(exist_ok=True)

    # setup full screen or window mode
    if config.FLLSCRN:
        current_rez = (pg.display.Info().current_w, pg.display.Info().current_h)
        screen = pg.display.set_mode(current_rez, pg.SCALED | pg.NOFRAME | pg.FULLSCREEN, vsync=config.VSYNC)
    else:
        screen = pg.display.set_mode((config.WIDTH, config.HEIGHT), pg.SCALED, vsync=config.VSYNC)

    cur_w, cur_h = screen.get_size()
    screen_size = (cur_w, cur_h)

    colony = pg.sprite.Group()
    world = PheromoneGrid(screen_size, config)

    malicious_ants_num = 0
    worker_ants_num = 0
    for n in range(config.ANTS):
        if n < config.ANTS * config.MAL_ANT_FRC:
            colony.add(MaliciousAnt(screen, world, config))
            malicious_ants_num += 1
        else:
            colony.add(WorkerAnt(screen, world, config))
            worker_ants_num += 1
    clock = pg.time.Clock()

    # main loop
    collected_food_list = []
    delivered_food_list = []
    ants_collected_ever_list = []
    ants_delivered_ever_list = []

    sim_step = 0
    while sim_step < config.N:
        sim_step += 1
        for e in pg.event.get():
            if e.type == pg.QUIT or e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE:
                return

        dt = clock.tick(config.FPS) / 1000
        pheroImg = world.update(dt)
        colony.update(dt)
        collected_food = sum(ant.collected_food for ant in colony.sprites() if isinstance(ant, WorkerAnt))
        delivered_food = sum(ant.delivered_food for ant in colony.sprites() if isinstance(ant, WorkerAnt))
        ants_collected_ever = 0
        ants_delivered_ever = 0
        for ant in colony.sprites():
            if isinstance(ant, WorkerAnt) and ant.delivered_food != 0:
                ants_delivered_ever += 1
            if isinstance(ant, WorkerAnt) and ant.collected_food != 0:
                ants_collected_ever += 1

        screen.fill(0)  # fill MUST be after sensors update, so previous draw is visible to them
        rescaled_img = pg.transform.scale(pheroImg, screen_size)
        pg.Surface.blit(screen, rescaled_img, (0, 0))
        pg.draw.circle(screen, config.HOME_COLOR, config.L_NEST, config.R_NEST / config.PRATIO)
        pg.draw.circle(screen, config.FOOD_COLOR, config.L_FOOD, config.R_FOOD / config.PRATIO)

        if result_directory is None:
            font = pg.font.Font(None, 30)
            screen.blit(font.render(f"simulation step: {sim_step}", True, [0, 200, 0]), (8, 8))
            screen.blit(font.render(f"total collected foods:{collected_food}", True, [0, 200, 0]), (8, 30))
            screen.blit(font.render(f"total delivered foods:{delivered_food}", True, [0, 200, 0]), (8, 60))
            screen.blit(font.render(f"ants ever collected:{ants_collected_ever}", True, [0, 200, 0]), (8, 90))
            screen.blit(font.render(f"ants ever delivered:{ants_delivered_ever}", True, [0, 200, 0]), (8, 120))

        if config.SHOW_COLONY:
            colony.draw(screen)

        ants_collected_ever_list.append(ants_collected_ever)
        ants_delivered_ever_list.append(ants_delivered_ever)
        collected_food_list.append(collected_food)
        delivered_food_list.append(delivered_food)

        if result_directory is not None:
            if sim_step % 2000 == 0:
                pg.image.save(screen, str(image_directory) + f"\\{sim_step}.png")

        pg.display.update()

    return collected_food_list, delivered_food_list, ants_collected_ever_list, ants_delivered_ever_list
