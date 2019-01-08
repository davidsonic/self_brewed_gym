import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import spaces
# debug
from mujoco_py import functions
import time



class MyInvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'my_inverted_pendulum.xml', 2)

        #debug, adv_action_space is 6 dimensional
        self._adv_f_bname='pole'
        bnames = self.sim.model.body_names
        self._adv_bindex = bnames.index(self._adv_f_bname)
        # adv force range, TODO: update adv_max_force
        adv_max_force=3.0
        high_adv = np.ones(6) * adv_max_force
        low_adv = - high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)
        self.pro_action_space = self.action_space

        #target0, target1 pos
        self.target0_site_id = self.model.site_name2id('target0')
        self.target1_site_id = self.model.site_name2id('target1')
        self.tendon_id = self.model.tendon_name2id('tendon_force')
        # self.view_counter=0



    # debug
    def applyForce(self, advact):
        # print('applied force', advact)
        adv_act = advact.copy().astype(np.float64)
        idx = self._adv_bindex
        nv = self.model.nv
        qfrc_applied= np.zeros((nv), dtype=np.float64)
        self.force = adv_act[:3]
        self.torque = adv_act[3:]
        self.point = self.model.site_pos[self.target0_site_id]
        # self.point = np.array([0, 0 ,0], dtype=np.float64)
        functions.mj_applyFT(self.model, self.data, self.force, self.torque, self.point, idx, qfrc_applied)
        self.data.qfrc_applied[:] = qfrc_applied
        #calculate target1 pos based on force and torque
        mag_force=np.linalg.norm(self.force)
        target1_loc = self.point + mag_force * self.torque

        #visualize force
        # 0. Enable showing
        self.model.tendon_rgba[self.tendon_id][3] = 1
        # 1. change target0 pos to point
        self.model.site_pos[self.target0_site_id] = self.point
        # 2. change target1 pos
        self.model.site_pos[self.target1_site_id] = target1_loc




    # seperate do_simulation with applied force
    def step(self, a):
        reward = 1.0
        # debug
        if hasattr(a, 'pro'):
            pro_act = a.pro
            self.do_simulation(pro_act, self.frame_skip)
            # debug apply perturb force
            if a.adv.any():
                self.applyForce(a.adv)
            else:
                # hide tendon
                self.model.tendon_rgba[self.tendon_id][3]=0
        else:
            self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not notdone
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    #debug
    def sample_action(self):
        class act(object):
            def __init__(self, pro=None, adv=None):
                self.pro=pro
                self.adv=adv

        sa = act(self.pro_action_space.sample(), self.adv_action_space.sample())
        return sa


    #debug
    def _render_callback(self):
        pass
        # sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        # site_id  = self.model.site_name2id('target1')
        # self.model.site_pos[site_id] = self.goal - sites_offset[0]
        # self.sim.forward()

