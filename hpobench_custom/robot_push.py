# https://github.com/zi-w/Ensemble-Bayesian-Optimization/blob/master/test_functions/push_function.py
from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.util import rng_helper

import ConfigSpace as CS
import time
from typing import Union, Dict

import numpy as np
import pygame
from Box2D import *


class guiWorld:
    def __init__(self, fps):
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 1000, 1000
        self.TARGET_FPS = fps
        self.PPM = 10.0  # pixels per meter
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), 0, 32)
        pygame.display.set_caption('push simulator')
        self.clock = pygame.time.Clock()
        self.screen_origin = b2Vec2(self.SCREEN_WIDTH / (2 * self.PPM), self.SCREEN_HEIGHT / (self.PPM * 2))
        self.colors = {
            b2_staticBody: (255, 255, 255, 255),
            b2_dynamicBody: (163, 209, 224, 255)
        }

    def draw(self, bodies, bg_color=(64, 64, 64, 0)):
        def my_draw_polygon(polygon, body, fixture):
            vertices = [(self.screen_origin + body.transform * v) * self.PPM for v in polygon.vertices]
            vertices = [(v[0], self.SCREEN_HEIGHT - v[1]) for v in vertices]
            color = self.colors[body.type]
            if body.userData == "obs":
                color = (123, 128, 120, 0)
            if body.userData == "hand":
                color = (174, 136, 218, 0)

            pygame.draw.polygon(self.screen, color, vertices)

        def my_draw_circle(circle, body, fixture):
            position = (self.screen_origin + body.transform * circle.pos) * self.PPM
            position = (position[0], self.SCREEN_HEIGHT - position[1])
            color = self.colors[body.type]
            if body.userData == "hand":
                color = (174, 136, 218, 0)
            pygame.draw.circle(self.screen, color, [int(x) for x in
                                                    position], int(circle.radius * self.PPM))

        b2PolygonShape.draw = my_draw_polygon
        b2CircleShape.draw = my_draw_circle
        # draw the world
        self.screen.fill(bg_color)
        self.clock.tick(self.TARGET_FPS)
        for body in bodies:
            for fixture in body.fixtures:
                fixture.shape.draw(body, fixture)
        pygame.display.flip()


# this is the interface to pybox2d
class b2WorldInterface:
    def __init__(self, do_gui=True):
        self.world = b2World(gravity=(0.0, 0.0), doSleep=True)
        self.do_gui = do_gui
        self.TARGET_FPS = 100
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.VEL_ITERS, self.POS_ITERS = 10, 10
        self.bodies = []

        if do_gui:
            self.gui_world = guiWorld(self.TARGET_FPS)
            # raw_input()
        else:
            self.gui_world = None

    def initialize_gui(self):
        if self.gui_world == None:
            self.gui_world = guiWorld(self.TARGET_FPS)
        self.do_gui = True

    def stop_gui(self):
        self.do_gui = False

    def add_bodies(self, new_bodies):
        """ add a single b2Body or list of b2Bodies to the world"""
        if type(new_bodies) == list:
            self.bodies += new_bodies
        else:
            self.bodies.append(new_bodies)

    def step(self, show_display=True, idx=0):
        self.world.Step(self.TIME_STEP, self.VEL_ITERS, self.POS_ITERS)
        if show_display and self.do_gui:
            self.gui_world.draw(self.bodies)


class end_effector:
    def __init__(self, b2world_interface, init_pos, base, init_angle, hand_shape='rectangle', hand_size=(0.3, 1)):
        world = b2world_interface.world
        self.hand = world.CreateDynamicBody(position=init_pos, angle=init_angle)
        self.hand_shape = hand_shape
        self.hand_size = hand_size
        # forceunit for circle and rect
        if hand_shape == 'rectangle':
            rshape = b2PolygonShape(box=hand_size)
            self.forceunit = 30.0
        elif hand_shape == 'circle':
            rshape = b2CircleShape(radius=hand_size)
            self.forceunit = 100.0
        elif hand_shape == 'polygon':
            rshape = b2PolygonShape(vertices=hand_size)
        else:
            raise Exception("%s is not a correct shape" % hand_shape)

        self.hand.CreateFixture(
            shape=rshape,
            density=.1,
            friction=.1
        )
        self.hand.userData = "hand"

        friction_joint = world.CreateFrictionJoint(
            bodyA=base,
            bodyB=self.hand,
            maxForce=2,
            maxTorque=2,
        )
        b2world_interface.add_bodies(self.hand)

    def set_pos(self, pos, angle):
        self.hand.position = pos
        self.hand.angle = angle

    def apply_wrench(self, rlvel=(0, 0), ravel=0):

        avel = self.hand.angularVelocity
        delta_avel = ravel - avel
        torque = self.hand.mass * delta_avel * 30.0
        self.hand.ApplyTorque(torque, wake=True)

        lvel = self.hand.linearVelocity
        delta_lvel = b2Vec2(rlvel) - b2Vec2(lvel)
        force = self.hand.mass * delta_lvel * self.forceunit
        self.hand.ApplyForce(force, self.hand.position, wake=True)

    def get_state(self, verbose=False):
        state = list(self.hand.position) + [self.hand.angle] + \
                list(self.hand.linearVelocity) + [self.hand.angularVelocity]
        if verbose:
            print_state = ["%.3f" % x for x in state]
            print
            "position, velocity: (%s), (%s) " % \
            ((", ").join(print_state[:3]), (", ").join(print_state[3:]))

        return state


def create_body(base, b2world_interface, body_shape, body_size, body_friction, body_density, obj_loc):
    world = b2world_interface.world

    link = world.CreateDynamicBody(position=obj_loc)
    if body_shape == 'rectangle':
        linkshape = b2PolygonShape(box=body_size)
    elif body_shape == 'circle':
        linkshape = b2CircleShape(radius=body_size)
    elif body_shape == 'polygon':
        linkshape = b2PolygonShape(vertices=body_size)
    else:
        raise Exception("%s is not a correct shape" % body_shape)

    link.CreateFixture(
        shape=linkshape,
        density=body_density,
        friction=body_friction,
    )
    friction_joint = world.CreateFrictionJoint(
        bodyA=base,
        bodyB=link,
        maxForce=5,
        maxTorque=2,
    )

    b2world_interface.add_bodies([link])
    return link


def make_base(table_width, table_length, b2world_interface):
    world = b2world_interface.world
    base = world.CreateStaticBody(
        position=(0, 0),
        shapes=b2PolygonShape(box=(table_length, table_width)),
    )

    b2world_interface.add_bodies([base])
    return base


def run_simulation(world, body, body2, robot, robot2, xvel, yvel, \
                   xvel2, yvel2, rtor, rtor2, simulation_steps,
                   simulation_steps2, noise_level=1e-6):
    # simulating push with fixed direction pointing from robot location to body location
    desired_vel = np.array([xvel, yvel])
    rvel = b2Vec2(desired_vel[0] + np.random.normal(0, noise_level), desired_vel[1] + np.random.normal(0, noise_level))

    desired_vel2 = np.array([xvel2, yvel2])
    rvel2 = b2Vec2(desired_vel2[0] + np.random.normal(0, noise_level), desired_vel2[1] + np.random.normal(0, noise_level))

    tmax = np.max([simulation_steps, simulation_steps2])
    for t in range(tmax + 100):
        if t < simulation_steps:
            robot.apply_wrench(rvel, rtor)
        if t < simulation_steps2:
            robot2.apply_wrench(rvel2, rtor2)
        world.step()

    return (list(body.position), list(body2.position))


class RobotPushBench(AbstractBenchmark):
    def __init__(self, rng: Union[int, np.random.RandomState, None] = None):
        super(RobotPushBench, self).__init__()

        self.rng = rng_helper.get_rng(rng=rng)
        np.random.seed(0)
        self.env = b2WorldInterface(False)
        self.num_dims = 14
        self.noise_level = 1e-6

        # starting xy locations for the two objects
        self.sxy = (0, 2)
        self.sxy2 = (0, -2)
        # goal xy locations for the two objects
        self.gxy = [4, 3.5]
        self.gxy2 = [-4, 3.5]

        self.defaults = {"rx": 0.,
                         "ry": 0.,
                         "xvel": 0.,
                         "y_vel": 0.,
                         "simu_steps_base": 16,
                         "init_angle": np.pi,
                         "rx2": 0.,
                         "rv2": 0.,
                         "xvel2": 0.,
                         "y_vel2": 0.,
                         "simu_steps2_base": 16,
                         "init_angle2": np.pi,
                         "rtor": 0.,
                         "rtor2": 0.,
                         }

    @staticmethod
    def get_configuration_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """
        Get the configuration space for this benchmark
        Parameters
        ----------
        seed : int, None
            Random seed for the configuration space.

        Returns
        -------
        CS.ConfigurationSpace -
            Containing the benchmark's hyperparameter
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        cs = CS.ConfigurationSpace(seed=seed)
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter(name='rx', lower=-5., upper=5., default_value=0.),
            CS.UniformFloatHyperparameter(name='ry', lower=-5, upper=5, default_value=0.),
            CS.UniformFloatHyperparameter(name='xvel', lower=-10, upper=10., default_value=0.),
            CS.UniformFloatHyperparameter(name='yvel', lower=-10., upper=10., default_value=0.),
            CS.UniformIntegerHyperparameter(name='simu_steps_base', lower=2, upper=30, default_value=16),
            CS.UniformFloatHyperparameter(name='init_angle', lower=0., upper=2*np.pi, default_value=np.pi),
            CS.UniformFloatHyperparameter(name='rx2', lower=-5., upper=5., default_value=0.),
            CS.UniformFloatHyperparameter(name='ry2', lower=-5., upper=5., default_value=0.),
            CS.UniformFloatHyperparameter(name='xvel2', lower=-10., upper=10., default_value=0.),
            CS.UniformFloatHyperparameter(name='yvel2', lower=-10., upper=10., default_value=0.),
            CS.UniformIntegerHyperparameter(name='simu_steps2_base', lower=2, upper=30, default_value=16),
            CS.UniformFloatHyperparameter(name='init_angle2', lower=0, upper=2*np.pi, default_value=np.pi),
            CS.UniformFloatHyperparameter(name='rtor', lower=-5., upper=5., default_value=0.),
            CS.UniformFloatHyperparameter(name='rtor2', lower=-5., upper=5., default_value=0.)
        ])
        return cs

    @staticmethod
    def get_fidelity_space(seed: Union[int, None] = None) -> CS.ConfigurationSpace:
        """ Defines the available fidelity parameters as a "fidelity space" for each benchmark.
        Parameters
        ----------
        seed: int, None
            Seed for the fidelity space.
        Returns
        -------
        ConfigSpace.ConfigurationSpace
            A valid configuration space for the benchmark's fidelity parameters
        """
        seed = seed if seed is not None else np.random.randint(1, 100000)
        fidel_space = CS.ConfigurationSpace(seed=seed)

        fidel_space.add_hyperparameters([
            CS.Constant("constant_fidelity", 1.0),
        ])
        return fidel_space

    @property
    def f_max(self):
        # maximum value of this function
        return np.linalg.norm(np.array(self.gxy) - np.array(self.sxy)) \
               + np.linalg.norm(np.array(self.gxy2) - np.array(self.sxy2))

    @AbstractBenchmark.check_parameters
    def objective_function(self, configuration: Union[Dict, CS.Configuration],
                           fidelity: Union[Dict, CS.Configuration, None] = None,
                           rng: Union[np.random.RandomState, int, None] = None,
                           **kwargs) -> Dict:
        """
        Use a heuristic controller to control a lunar lander

        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : inverse of the final reward
            cost : time to run all agents
            info : Dict
                max_episodes : the maximum length of an episode
                budget : number of agents used
                all_runs : the episode length of all runs of all agents
                fidelity : the used fidelities in this evaluation
        """
        self.rng = rng_helper.get_rng(rng=rng, self_rng=self.rng)
        np.random.seed(self.rng.randint(1, 100000))

        # fill in missing entries with default values for 'incomplete/reduced' configspaces
        new_config = self.defaults
        new_config.update(configuration)
        configuration = new_config

        start_time = time.time()


        # returns the reward of pushing two objects with two robots
        rx = configuration['rx']
        ry = configuration['ry']
        xvel = configuration['xvel']
        yvel = configuration['yvel']
        simu_steps = configuration['simu_steps_base'] * 10
        init_angle = configuration['init_angle']
        rx2 = configuration['rx2']
        ry2 = configuration['ry2']
        xvel2 = configuration['xvel2']
        yvel2 = configuration['y_vel2']
        simu_steps2 = configuration['simu_steps2_base'] * 10
        init_angle2 = configuration['init_angle2']
        rtor = configuration['rtor']
        rtor2 = configuration['rtor2']

        initial_dist = self.f_max

        world = b2WorldInterface(False)
        oshape, osize, ofriction, odensity, bfriction, hand_shape, hand_size = \
            'circle', 1, 0.01, 0.05, 0.01, 'rectangle', (1, 0.3)

        base = make_base(500, 500, world)
        body = create_body(base, world, 'rectangle', (0.5, 0.5), ofriction, odensity, self.sxy)
        body2 = create_body(base, world, 'circle', 1, ofriction, odensity, self.sxy2)

        robot = end_effector(world, (rx, ry), base, init_angle, hand_shape, hand_size)
        robot2 = end_effector(world, (rx2, ry2), base, init_angle2, hand_shape, hand_size)
        (ret1, ret2) = run_simulation(world, body, body2, robot, robot2, xvel, yvel, \
                                      xvel2, yvel2, rtor, rtor2, simu_steps, simu_steps2)

        ret1 = np.linalg.norm(np.array(self.gxy) - ret1)
        ret2 = np.linalg.norm(np.array(self.gxy2) - ret2)

        cost = time.time() - start_time

        total_reward = initial_dist - ret1 - ret2

        return {'function_value': total_reward,
                'cost': cost,
                'info': {'fidelity': fidelity}
                }

    @AbstractBenchmark.check_parameters
    def objective_function_test(self, configuration: Union[Dict, CS.Configuration],
                                fidelity: Union[Dict, CS.Configuration, None] = None,
                                rng: Union[np.random.RandomState, int, None] = None,
                                **kwargs) -> Dict:
        """
        Validate a configuration on the cartpole benchmark. Use the full budget.
        Parameters
        ----------
        configuration : Dict, CS.Configuration
        fidelity: Dict, None
            Fidelity parameters, check get_fidelity_space(). Uses default (max) value if None.
        rng : np.random.RandomState, int, None
            Random seed to use in the benchmark. To prevent overfitting on a single seed, it is possible to pass a
            parameter ``rng`` as 'int' or 'np.random.RandomState' to this function.
            If this parameter is not given, the default random state is used.
        kwargs

        Returns
        -------
        Dict -
            function_value : average episode length
            cost : time to run all agents
            info : Dict
                max_episodes : the maximum length of an episode
                budget : number of agents used
                all_runs : the episode length of all runs of all agents
                fidelity : the used fidelities in this evaluation
        """

        return self.objective_function(configuration=configuration, fidelity=fidelity, rng=rng,
                                       **kwargs)

    @staticmethod
    def get_meta_information() -> Dict:
        return {'name': 'RobotPush',
                'references': ['@InProceedings{WangGKJ-aistats18,'
                               'title = {Batched Large-scale Bayesian Optimization in High-dimensional Spaces},'
                               'url = {http://proceedings.mlr.press/v84/wang18c.html},'
                               'author = {Zi Wang and Clement Gehring and Pushmeet Kohli and Stefanie Jegelka},'
                               'booktitle = {International Conference on Artificial Intelligence and Statistics,'
                               '{AISTATS} 2018, 9-11 April 2018, Playa Blanca, Lanzarote, Canary Islands, Spain},'
                               'pages = {745--754},'
                               'year      = {2018}},'
                               'publisher = {{PMLR}}'],
                'url': "https://github.com/zi-w/Ensemble-Bayesian-Optimization"
                }