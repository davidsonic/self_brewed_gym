from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
# debug
from gym import spaces

class HumanoidStandEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):

        mujoco_env.MujocoEnv.__init__(self, 'humanoidstandup.xml', 5)
        utils.EzPickle.__init__(self)

        # debug
        self._adv_f_bname = 'left_upper_arm'
        bnames = self.sim.model.body_names
        self._adv_bindex = bnames.index(self._adv_f_bname)
        adv_max_force = 5.0
        high_adv = np.ones(1) * adv_max_force
        low_adv = - high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space

    def _get_obs(self):
        data = self.sim.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        pos_after = self.sim.data.qpos[2]
        data = self.sim.data
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

        done = bool(False)
        return self._get_obs(), reward, done, dict(reward_linup=uph_cost, reward_quadctrl=-quad_ctrl_cost, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 0.8925
        self.viewer.cam.elevation = -20

    # debug
    def sample_action(self):
        class act(object):
            def __init__(self, pro=None, adv=None):
                self.pro=pro
                self.adv=adv

        sa= act(self.pro_action_space.sample(), self.adv_action_space.sample())
        return sa