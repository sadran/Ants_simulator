import numpy as np


class Config:
    #################### Environment setyp

    FLLSCRN = True             # True for Full screen, or False for Window
    ANTS = 1024                   # Number of Ants in simulation
    N = 50000                    # number of simulation steps per experiment
    WIDTH = 1920                # default 1200
    HEIGHT = 1080               # default 800
    PRATIO = 4                  # Pixel Size for Pheromone grid
    L_FOOD = (372, 36)          # food location
    R_FOOD = 16                 # food source radius
    L_NEST = (960, 540)         # nest location
    R_NEST = 20                 # nest radius

    ###################### Hyper parameters
    MAX_SPEED = 50              # maximum ants speed
    THETA_MAX = 0.8 * np.pi     # range of vector generation for direction selection
    L_MAX = 40                  # maximum magnitude of vector generation for direction selection
    MU = 0.1 * np.pi            # coefficient for random noise
    LAMBDA = 0.01               # coefficient for pheromone intensity
    POLICY = 0
    K = 1                       # Evaporation rate
    P_MAX = 1000                 # Maximum patience
    TP = 5
    X = 32                      # number of random vectors generated for direction selection
    FPS = 60                    # 48-90
    MAL_ANT_FRC = .03
    VSYNC = True                # limit frame rate to refresh rate

    ##################### Colors
    HOME_COLOR = (125, 0, 255)
    FOOD_COLOR = [0, 255, 0]
    HOME_PHEROMONE_COLOR = (0, 0, 255)
    FOOD_PHEROMONE_COLOR = (0, 255, 0)
    MIS_PHEROMONE_COLOR = (225, 0, 0)
    CAUT_PHEROMONE_COLOR = (255, 0, 250)
    MAX_PHEROMONE_INTENSITY = 1000

    ######################### Visualization options
    SHOW_HOME_PHERO = False
    SHOW_FOOD_PHERO = True
    SHOW_MIS_PHERO = True
    SHOW_CAUT_PHERO = True
    SHOW_COLONY = False    # if True, it visualizes the ants as well
