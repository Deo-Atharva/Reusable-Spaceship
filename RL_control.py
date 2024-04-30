import krpc
import time
import numpy as np
import math

from pynput.keyboard import Listener, Key, KeyCode
from collections import defaultdict
from enum import Enum
import subprocess
import os
import queue

import threading
import traceback
import pickle
import usb

import cv2
from mss import mss

from base_method import *

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

# Rocket Build: FLT800,mk1 command pod,mk16 parachute, airbrakes, rv105 thruster, delta deluxe winglet,lvt30 fuel,lt2 landing strut
# Cheat menu right shift+F12

driving_agent = 'RL'
hdir = '/home/xpsucla/Spacecraft/Kerbal Space Program/Python files/Python Data/'
version = 1
wind_sit = '/No Wind/'
date_dir = '/141123/'
save_dir = hdir + driving_agent + wind_sit + date_dir + 'v' + str(version)
home_dir = hdir + driving_agent + wind_sit + date_dir
fname_external = "External_Video_" + driving_agent + "_v" + str(version) + ".avi"

HIDDEN_UNITS_1 = 4
HIDDEN_UNITS_2 = 4
ACTION_DIM = 7
LEARNING_RATE = 5e-5
GAMMA = 0.99
NUM_EPS = 3500
num_inp = 6
EXP_END = 0
PENALTY = 1000

epss = 2866

mdir = hdir + driving_agent + wind_sit + date_dir + '/Inference/v1_RL_lr_' + str(LEARNING_RATE) + '_Gamma_' + str(GAMMA) + '_itr_'
mdir_dat = hdir + driving_agent + wind_sit + date_dir + '/Inf Data/v1_RL_lr_' + str(LEARNING_RATE) + '_Gamma_' + str(GAMMA) + '_itr_'
mdir_epload = mdir + str(epss)

class Ctrl(Enum):
    (
        QUIT,
        TAKEOFF,
        LANDING,
        MOVE_LEFT,
        MOVE_RIGHT,
        MOVE_FORWARD,
        MOVE_BACKWARD,
        MOVE_UP,
        MOVE_DOWN,
        TURN_LEFT,
        TURN_RIGHT,
        START_EXP,
        END_EXP,
        TAKE_REF,
        START_WIND
    ) = range(15)


QWERTY_CTRL_KEYS = {
    Ctrl.QUIT: Key.esc,
    Ctrl.TAKEOFF: "t",
    Ctrl.LANDING: "l",
    Ctrl.MOVE_LEFT: "a",
    Ctrl.MOVE_RIGHT: "d",
    Ctrl.MOVE_FORWARD: "w",
    Ctrl.MOVE_BACKWARD: "s",
    Ctrl.MOVE_UP: Key.up,
    Ctrl.MOVE_DOWN: Key.down,
    Ctrl.TURN_LEFT: Key.left,
    Ctrl.TURN_RIGHT: Key.right,
    Ctrl.START_EXP: "o",
    Ctrl.END_EXP: "p",
    Ctrl.TAKE_REF: "r",
    Ctrl.START_WIND: "b"
}

AZERTY_CTRL_KEYS = QWERTY_CTRL_KEYS.copy()
AZERTY_CTRL_KEYS.update(
    {
        Ctrl.MOVE_LEFT: "q",
        Ctrl.MOVE_RIGHT: "d",
        Ctrl.MOVE_FORWARD: "z",
        Ctrl.MOVE_BACKWARD: "s",
    }
)

## KEYBOARD CLASS
class KeyboardCtrl(Listener):
    def __init__(self, ctrl_keys=None):
        self._ctrl_keys = self._get_ctrl_keys(ctrl_keys)
        self._key_pressed = defaultdict(lambda: False)
        self._last_action_ts = defaultdict(lambda: 0.0)
        super().__init__(on_press=self._on_press, on_release=self._on_release)
        self.start()

    def _on_press(self, key):
        if isinstance(key, KeyCode):
            self._key_pressed[key.char] = True
        elif isinstance(key, Key):
            self._key_pressed[key] = True
        if self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]:
            return False
        else:
            return True

    def _on_release(self, key):
        if isinstance(key, KeyCode):
            self._key_pressed[key.char] = False
        elif isinstance(key, Key):
            self._key_pressed[key] = False
        return True

    def quit(self):
        return not self.running or self._key_pressed[self._ctrl_keys[Ctrl.QUIT]]

    def _axis(self, left_key, right_key):
        diff = int(self._key_pressed[right_key]) - int(self._key_pressed[left_key])
        if (diff>0):
            return '01'
        elif (diff<0):
            return '10'
        else:
            return '00'

    def yaw(self):
        return self._axis(
            self._ctrl_keys[Ctrl.MOVE_LEFT],
            self._ctrl_keys[Ctrl.MOVE_RIGHT]
        )

    def pitch(self):
        return self._axis(
            self._ctrl_keys[Ctrl.MOVE_BACKWARD],
            self._ctrl_keys[Ctrl.MOVE_FORWARD]
        )

    def roll(self):
        return self._axis(
            self._ctrl_keys[Ctrl.TURN_LEFT],
            self._ctrl_keys[Ctrl.TURN_RIGHT]
        )

    def thrttle(self):
        return self._axis(
            self._ctrl_keys[Ctrl.MOVE_DOWN],
            self._ctrl_keys[Ctrl.MOVE_UP]
        )


    def has_piloting_cmd(self):
        return (
            bool(self.roll())
            or bool(self.pitch())
            or bool(self.yaw())
            or bool(self.thrttle())
        )

    def _rate_limit_cmd(self, ctrl, delay):
        now = time.time()
        if self._last_action_ts[ctrl] > (now - delay):
            return str(1)
        elif self._key_pressed[self._ctrl_keys[ctrl]]:
            self._last_action_ts[ctrl] = now
            return str(1)
        else:
            return str(0)

    def takeoff(self):
        return self._rate_limit_cmd(Ctrl.TAKEOFF, 2.0)

    def landing(self):
        return self._rate_limit_cmd(Ctrl.LANDING, 2.0)

    def take_reference(self):
        return self._rate_limit_cmd(Ctrl.TAKE_REF, 2.0)

    def start_experiment(self):
        return self._rate_limit_cmd(Ctrl.START_EXP, 2.0)

    def end_experiment(self):
        return self._rate_limit_cmd(Ctrl.END_EXP, 2.0)
    
    def start_wind(self):
        return self._rate_limit_cmd(Ctrl.START_WIND, 2.0)

    def _get_ctrl_keys(self, ctrl_keys):
        # Get the default ctrl keys based on the current keyboard layout:
        if ctrl_keys is None:
            ctrl_keys = QWERTY_CTRL_KEYS
            try:
                # Olympe currently only support Linux
                # and the following only works on *nix/X11...
                keyboard_variant = (
                    subprocess.check_output(
                        "setxkbmap -query | grep 'variant:'|"
                        "cut -d ':' -f2 | tr -d ' '",
                        shell=True,
                    )
                    .decode()
                    .strip()
                )
            except subprocess.CalledProcessError:
                pass
            else:
                if keyboard_variant == "azerty":
                    ctrl_keys = AZERTY_CTRL_KEYS
        return ctrl_keys 

#RL Policy Class

class PolicyNet(keras.Model):
    def __init__(self, action_dim= ACTION_DIM):
        super(PolicyNet, self).__init__()
        # self.fc1 = layers.Dense(HIDDEN_UNITS_1, activation="relu", input_shape=(1,num_inp),\
        #     kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),bias_initializer=initializers.Zeros())
        self.fc1 = layers.Dense(action_dim, activation="softmax", input_shape=(1,num_inp),\
            kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),bias_initializer=initializers.Zeros())
        # self.bn1 = layers.BatchNormalization()
        # self.fc2 = layers.Dense(HIDDEN_UNITS_2, activation="relu",kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01)\
        #     ,bias_initializer=initializers.Zeros())
        # self.fc3 = layers.Dense(action_dim,activation="softmax"\
        #     ,kernel_initializer=initializers.RandomUniform(minval=-0.01, maxval=0.01),bias_initializer=initializers.Zeros())
    
    def call(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x

    def process(self, observations):
        # Process batch observations using `call(x)`
        # behind-the-scenes
        action_probabilities = self.predict_on_batch(observations)
        return action_probabilities#np.clip(action_probabilities,1e-7,1-1e-7)

class Agent(object):
    def __init__(self, action_dim=ACTION_DIM):
        """Agent with a neural-network brain powered
        policy
        Args:
        action_dim (int): Action dimension
        """
        self.policy_net = PolicyNet(action_dim=action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-8)
        self.gamma = GAMMA

    def policy(self, observation):
        observation = observation.reshape(1, num_inp)
        observation = tf.convert_to_tensor(observation,dtype=tf.float32)
        # print(observation)
        action_logits = self.policy_net(observation)
        # print('Action: ',action_logits)
        action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
        # print('Action: ',action_logits)
        return action

    def get_action(self, observation):
        action = self.policy(observation).numpy()
        # print(action)
        return action.squeeze()

    def learn(self, states, rewards, actions):
        discounted_reward = 0
        discounted_rewards = []
        # print(rewards)
        rewards.reverse()
        #print(self.policy_net.trainable_variables)
        for r in rewards:
            discounted_reward = r + self.gamma * discounted_reward
            discounted_rewards.append(discounted_reward)
        discounted_rewards.reverse()
        # print(discounted_rewards)
        discounted_rewards = list(np.array(discounted_rewards) - np.mean(np.array(discounted_rewards)))
        # print(discounted_rewards)
        for state, reward, action in zip(states,discounted_rewards, actions):
            with tf.GradientTape() as tape:
                action_probabilities = tf.clip_by_value(self.policy_net(np.array([state]),training=True), clip_value_min=1e-2, clip_value_max=1-1e-2)
                # action_probabilities = self.policy_net(np.array([state]),training=True)
                # print(action_probabilities)
                loss = self.loss(action_probabilities,action, reward)
                grads = tape.gradient(loss,self.policy_net.trainable_variables)
                self.optimizer.apply_gradients(zip(grads,self.policy_net.trainable_variables))
        # print(self.policy_net.trainable_variables)

    def loss(self, action_probabilities, action, reward):
        # log_prob = tf.math.log(action_probabilities(action))
        dist = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        # print(tf.math.log(action_probabilities))
        # print(action)
        # print(log_prob)
        loss = -log_prob * reward
        # print(reward)
        # print(loss)
        return loss

def reward_scheme(states):
    # s=0
    # if(states[0]+states[1]<=30):
    #     s = s + 60/(states[0]+states[1]+1)
    # elif(states[2]+states[3]<=30):
    #     s = s + 60/(states[2]+states[3]+1)
    # elif(states[4]+states[5]<=30):
    #     s = s + 15/(states[4]+states[5]+1)
    # s = -0.1*np.sum(states)
    # print(np.any(np.abs(states) > 100))
    if (np.sum(states) < 120):
        s = 25/np.sqrt(np.sum(states)+1)
    # if (states[0]+states[1]+states[2]+states[3]<=80):
    #     s = 25/np.sqrt(np.sum(states[0]+states[1]+states[2]+states[3])+1)
    # if (states[2]+states[3]<=40):
    #     s = 25/np.sqrt(np.sum(states[2]+states[3])+1)
    else:
        s = 0
    # elif (np.sum(states) < 100):
    #     s = 750/np.sqrt(np.sum(states)+1)
    # else:
    #     s =  -.1 * np.sum(states)
    # s =  np.sum(states)
    # if(np.any(np.abs(states) > 100) == False):
    #     s =  np.sum(states)
    # else:
    #     s =  np.array(PENALTY)
    # print('Rewards: ',s)
    return s

class RocketTracking(threading.Thread):
    def __init__(self):
        self.cntrl = KeyboardCtrl()
        self.save_file_onsite = save_dir
        self.conn = krpc.connect()
        self.vessel = self.conn.space_center.active_vessel
        self.camera_vessel = self.conn.space_center.camera
        # print(self.camera_vessel.focussed_vessel)
        # print(self.camera_view.max_distance)
        # print(self.camera_view.min_distance)
        # print(self.camera_view.distance)
        self.stop_processing = False
        self.state = np.zeros(4)
        self.state_all = np.zeros(6)
        self.vs = np.zeros(3)
        self.current_pos = np.zeros(4)
        self.initpos = np.zeros(3)
        # self.ref_frame_from = self.vessel.orbit.body.non_rotating_reference_frame
        self.ref_frame_from = self.vessel.orbit.body.reference_frame
        # self.ref_frame_to = self.vessel.orbital_reference_frame
        self.ref_frame_to = self.vessel.surface_reference_frame
        # self.ref_frame_to = self.vessel.surface_velocity_reference_frame
        self.us = np.array(self.conn.space_center.transform_direction(self.vessel.velocity(self.ref_frame_from),self.ref_frame_from,self.ref_frame_to))
        
        self.planet_radius = self.vessel.orbit.body.equatorial_radius
        self.vf = self.vessel.flight(self.ref_frame_from)
        self.lat_i = self.vessel.flight(self.ref_frame_from).latitude
        self.long_i = self.vessel.flight(self.ref_frame_from).longitude
        self.roll_i = self.vessel.flight(self.ref_frame_from).roll
        self.x_i = (math.pi/180) * self.planet_radius * self.vessel.flight(self.ref_frame_from).latitude
        self.y_i = (math.pi/180) * self.planet_radius * self.vessel.flight(self.ref_frame_from).longitude
        self.z_i = self.vessel.flight(self.ref_frame_from).mean_altitude
        self.max_alt = self.z_i

        self.loc_choices = np.array([-1.,-0.875,-0.75,0.75,0.875,1.])
        self.alt_choices = np.array([0.7,0.85,1.])
        self.alt_dir = np.array([-1,1])
        # self.epd = np.random.choice(self.loc_choices)
        
        self.xtarget_temp = np.array([self.lat_i+np.random.choice(self.loc_choices)*0.00075])
        self.ytarget_temp = np.array([self.long_i+np.random.choice(self.loc_choices)*0.00075])
        self.ztarget_temp = np.array([self.z_i+np.random.choice(self.alt_choices)*100])

        print('Latitude:',self.xtarget_temp[-1])
        print('Longitude:',self.ytarget_temp[-1])
        self.target_pos = np.array([self.xtarget_temp[-1], self.ytarget_temp[-1], self.ztarget_temp[-1],0.0])
        self.target_dev = self.target_pos - self.current_pos
        self.start_time = None
        self.ylim = -100
        self.reached_target = False
        # self.state_gains = [7,7,0.2,0.1]
        self.state_gains = [7,7,0.7,0.1]
        self.ts = time.time()
        self.wind = False
        self.twind = time.time()
        self.dtwind = 10
        self.wind_velocities = np.zeros(3)
        self.p_i = self.vessel.flight(self.ref_frame_from).pitch
        self.h_i = self.vessel.flight(self.ref_frame_from).heading
        # self.auto_p = self.vessel.auto_pilot
        # self.auto_p.engage()
        # self.auto_p.target_pitch_and_heading(90, 90)
        # self.auto_p.target_roll = 0



        self.chp_num = 0
        self.chp_num_d = 0
        # self.current_count = 0
        self.total_chp = self.xtarget_temp.shape[0]
        # print(self.total_chp)
        self.reached_checkpoint = False
        # self.z_num = self.local_targets.shape[0]
        self.dtol = [30,50,60,80,90,100,120,130,150,150]
        self.oi = self.vessel.rotation(self.vessel.surface_reference_frame)
        self.start_txt = 0
        self.start_wtxt = 0
        self.txt_time = time.time()
        self.wind_txt_time = time.time()
        self.dt_txt = 0.5
        self.way_draw = 0 
        self.final_checkpoint = False
        self.frame = None
        self.frame_rate_video = 10.0
        self.stop_video = False
        self.start_video = False
        self.fontFace = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1.0
        self.thickness = 2
        self.save_file_ext = home_dir + fname_external
        self.mid = 1
        self.chp_mid = 1
        # self.camera_vessel.mode = self.camera_vessel.mode.map
        self.s_d = [self.vessel.flight(self.ref_frame_from).latitude,self.vessel.flight(self.ref_frame_from).longitude,\
                           self.vessel.flight(self.ref_frame_from).mean_altitude]  

        self.show_states = False
        self.camera_vessel.heading = 0.0
        self.camera_vessel.pitch = 0.0
        # print(self.camera_vessel.pitch)
        self.camera_vessel.distance = 30.0
        self.vp = self.vessel.parts.root
        self.vw = np.zeros(3)
        self.f = np.zeros(3)
        self.kk = []
        self.vw_angle = 0.0
        self.pitch_angle = 0.0
        self.heading_angle = 0.0

        super().__init__()
        super().start()

    def start(self):
        pass
 
    def next_checkpoint(self):
        if(self.alt_r * (self.state[2]-math.copysign(5, self.state[2])) > 0 and self.chp_num<self.total_chp):
            time.sleep(0.1)
            self.chp_num += 1
            print('Next checkpoint: ',self.chp_num,'\n')
            print(self.state_all)

    def get_alt(self):
        self.alt_r = np.random.choice(self.alt_dir)
    
    def init_states(self):
        self.xtarget_temp = np.array([self.lat_i+np.random.choice(self.loc_choices)*0.00075])
        self.ytarget_temp = np.array([self.long_i+np.random.choice(self.loc_choices)*0.00075])
        self.alt_c = np.random.choice(self.alt_choices)*100
        self.ztarget_temp = np.array([self.z_i+(self.alt_c*self.alt_r)])

    def state_txt_display(self):
        if(time.time()-self.txt_time > self.dt_txt and self.show_states):
            if(self.start_txt !=0):
                self.txt.remove()
                self.ang_txt.remove()
            self.start_txt = 1
            # self.str_txt = '['+str(int(self.state[0]))+','+str(int(self.state[1]))+','+str(int(self.state[2]))+','+str(int(self.state[3]))+']'
            self.str_txt = '['+str(int(self.state[0]))+','+str(int(self.state[1]))+','+str(int(self.state[2]))+']'
            self.txt = self.conn.drawing.add_text(self.str_txt,self.vessel.surface_reference_frame,(20,40,40),self.oi)
            self.txt.size = 100
            self.txt.style = self.txt.style.bold

            self.pitch_angle = self.vessel.flight(self.ref_frame_from).pitch - self.p_i
            self.heading_angle = self.vessel.flight(self.ref_frame_from).heading - self.h_i
            self.str_angtxt = '[\u03B1: '+str(int(self.pitch_angle*10)/10)+'\u00b0, \u03B2: '+str(int(self.heading_angle*10)/10)+'\u00b0]'
            self.ang_txt = self.conn.drawing.add_text(self.str_angtxt,self.vessel.surface_reference_frame,(5,40,40),self.oi)
            self.ang_txt.size = 100
            self.ang_txt.style = self.ang_txt.style.bold
            self.txt_time = time.time()  
        elif(time.time()-self.txt_time > self.dt_txt and ~self.show_states):
            self.camera_vessel.distance = 30.0          

    def flight_trace(self):
        s = [self.vessel.flight(self.ref_frame_from).latitude,self.vessel.flight(self.ref_frame_from).longitude,\
                           self.vessel.flight(self.ref_frame_from).mean_altitude] 
        fl_p_1 = self.vessel.orbit.body.position_at_altitude(s[0],s[1],s[2], self.ref_frame_from)  
        fl_p_2 = self.vessel.orbit.body.position_at_altitude(self.s_d[0],self.s_d[1],self.s_d[2], self.ref_frame_from)  
        flt_1 = self.conn.drawing.add_line((fl_p_2[0],fl_p_2[1],fl_p_2[2]), (fl_p_1[0],fl_p_1[1],fl_p_1[2]), self.ref_frame_from)
        flt_1.color = (153./255,12./255,17./255)
        flt_1.thickness = 3
        self.s_d = s      

    def target_trace(self):
        if(self.chp_num==0):
            # s = [self.vessel.flight(self.ref_frame_from).latitude,self.vessel.flight(self.ref_frame_from).longitude,\
            #                self.vessel.flight(self.ref_frame_from).mean_altitude] 
            s = [self.lat_i,self.long_i,self.z_i]
            tt_1 = self.vessel.orbit.body.position_at_altitude(s[0],s[1],s[2], self.ref_frame_from)
            tt_2 = self.vessel.orbit.body.position_at_altitude(self.xtarget_temp[self.chp_num],self.ytarget_temp[self.chp_num]\
                                                            ,self.ztarget_temp[self.chp_num]+self.z_i, self.ref_frame_from)  
            tt_1 = np.array(tt_1)
            tt_2 = np.array(tt_2)
            ttt_1 = self.conn.drawing.add_line((tt_1[0],tt_1[1],tt_1[2]), (tt_2[0],tt_2[1],tt_2[2]), self.ref_frame_from)
            ttt_1.color = (250./255,12./255,246./255)
            ttt_1.thickness = 1
        else:
            tt_1 = self.vessel.orbit.body.position_at_altitude(self.xtarget_temp[self.chp_num-1],self.ytarget_temp[self.chp_num-1]\
                                                                ,self.ztarget_temp[self.chp_num-1]+self.z_i, self.ref_frame_from)  
            tt_2 = self.vessel.orbit.body.position_at_altitude(self.xtarget_temp[self.chp_num],self.ytarget_temp[self.chp_num]\
                                                            ,self.ztarget_temp[self.chp_num]+self.z_i, self.ref_frame_from)  
            tt_1 = np.array(tt_1)
            tt_2 = np.array(tt_2)
            ttt_1 = self.conn.drawing.add_line((tt_1[0],tt_1[1],tt_1[2]), (tt_2[0],tt_2[1],tt_2[2]), self.ref_frame_from)
            ttt_1.color = (250./255,12./255,246./255)
            ttt_1.thickness = 1


    def wind_var(self):
        rmin=0
        rmax=0
        # self.vw = np.zeros(3)
        if(self.chp_num >= 1 and self.chp_num < 3):
            rmin = 10000
            rmax = 25000
        elif(self.chp_num >= 3 and self.chp_num < 5):
            rmin = 15000
            rmax = 25000
        elif(self.chp_num == 5):
            rmin = 10000
            rmax = 15000
        elif(self.chp_num >= 6 and self.chp_num <= 10):
            rmin = 10000
            rmax = 15000
        # elif(self.chp_num == 10):
        #     rmin = 100000
        #     rmax = 125000
        if(self.chp_num != 0 and self.chp_num_d != self.chp_num and self.chp_num != 5):
            vx = np.random.choice(np.append(np.arange(-rmax,-rmin),np.arange(rmin,rmax)))/1000.0
            vy = np.random.choice(np.append(np.arange(-rmax,-rmin),np.arange(rmin,rmax)))/1000.0
            vz = np.random.choice(np.append(np.arange(-rmax,-rmin),np.arange(rmin,rmax)))/1000.0
            self.vw = np.array([vx,0,vz])
            print('Wind Speed:',self.vw)
        # elif(self.chp_num == 5):
        #     self.vw = np.array([0,0,0])
        #     print('Wind Speed:',self.vw)
        comrf = self.vp.center_of_mass_reference_frame
        com = self.vp.center_of_mass(comrf)
        vesvel = np.abs(np.array(self.vf.velocity))
        if(self.chp_num<5):
            k = np.abs(np.array(self.vf.aerodynamic_force)/(vesvel**2))
            # k = np.min(k)
            # k = np.array(self.vf.drag_coefficient)+np.array(self.vf.lift_coefficient)
            self.kk.append(k)
            # print(np.shape(np.array(self.kk)))
        else:
            k = np.mean(np.array(self.kk),axis=0)
        forcee = k*self.vw**2 + 2*k*vesvel*self.vw
        self.f = forcee
        if(self.chp_num!=0 and ~np.isnan(forcee[0]) and np.abs(vesvel[2])>10):
            self.vp.instantaneous_force(forcee,com,comrf)
        self.chp_num_d = self.chp_num

        # Display wind data
        vw_abs = np.linalg.norm(self.vw)
        self.vw_angle = np.arctan2(self.vw[2], self.vw[0]) * 180 / np.pi
        if(time.time()-self.wind_txt_time > self.dt_txt and self.show_states):
            if(self.start_wtxt !=0):
                self.wind_txt.remove()
                # self.wttt_1.remove()
            self.start_wtxt = 1
            # self.str_txt = '['+str(int(self.state[0]))+','+str(int(self.state[1]))+','+str(int(self.state[2]))+','+str(int(self.state[3]))+']'
            self.str_wtxt = '['+'|v|: '+str(int(vw_abs*10)/10)+', \u03A6: '+str(int(self.vw_angle*10)/10)+'\u00b0]'
            self.wind_txt = self.conn.drawing.add_text(self.str_wtxt,self.vessel.surface_reference_frame,(35,40,40),self.oi)
            self.wind_txt.size = 100
            self.wind_txt.style = self.wind_txt.style.bold
            self.wind_txt_time = time.time()
            # l = 1e-2
            # s = [self.vessel.flight(self.ref_frame_from).latitude,self.vessel.flight(self.ref_frame_from).longitude,\
            #                 self.vessel.flight(self.ref_frame_from).mean_altitude] 
            # tt_1 = self.vessel.orbit.body.position_at_altitude(s[0],s[1]+l*np.cos(self.vw_angle*np.pi/180),\
            #                                                 s[2]+l*np.sin(self.vw_angle*np.pi/180), self.ref_frame_from)
            # tt_2 = self.vessel.orbit.body.position_at_altitude(s[0],s[1]-l*np.cos(self.vw_angle*np.pi/180),\
            #                                                 s[2]-l*np.sin(self.vw_angle*np.pi/180), self.ref_frame_from)  
            # tt_1 = np.array(tt_1)
            # tt_2 = np.array(tt_2)
            # self.wttt_1 = self.conn.drawing.add_line((tt_1[0],tt_1[1],tt_1[2]), (tt_2[0],tt_2[1],tt_2[2]), self.ref_frame_from)
            # self.wttt_1.color = (250./255,12./255,246./255)
            # self.wttt_1.thickness = 1

    def state_divider(self):
        self.state_all[0] = self.state[0] if self.state[0]>=0 else 0
        self.state_all[1] = -self.state[0] if self.state[0]<0 else 0
        self.state_all[2] = self.state[1] if self.state[1]>=0 else 0
        self.state_all[3] = -self.state[1] if self.state[1]<0 else 0
        self.state_all[4] = self.state[2] if self.state[2]>=0 else 0
        self.state_all[5] = -self.state[2] if self.state[2]<0 else 0

    def reset_chp(self):
        self.chp_num = 0
        self.mid = 1

    def get_obs(self):
        self.current_pos[3] = self.vessel.flight(self.ref_frame_from).roll - self.roll_i
        pos_ar = self.vessel.velocity(self.ref_frame_from)
        self.vs = np.array(self.conn.space_center.transform_direction(pos_ar,self.ref_frame_from,self.ref_frame_to))
        self.ts = time.time()
        self.current_pos[0]= (math.pi/180) * self.planet_radius * (self.vessel.flight(self.ref_frame_from).latitude - self.lat_i)
        self.current_pos[1]= (math.pi/180) * self.planet_radius * (self.vessel.flight(self.ref_frame_from).longitude - self.long_i)
        self.current_pos[2]= self.vessel.flight(self.ref_frame_from).mean_altitude
        self.state[0] = -np.clip((self.current_pos[0]-((math.pi/180) * self.planet_radius * (self.xtarget_temp[self.chp_num] - self.lat_i)))*self.state_gains[0],-105,105)
        self.state[1] = np.clip((self.current_pos[1]-((math.pi/180) * self.planet_radius * (self.ytarget_temp[self.chp_num] - self.long_i)))*self.state_gains[1],-105,105)
        self.state[2] = np.clip((self.current_pos[2]-self.ztarget_temp[self.chp_num])*self.state_gains[2],-105,105)
        self.state[3] = np.clip(self.current_pos[3]*self.state_gains[3],-100,100)
        self.target_dev = self.target_pos - self.current_pos
        # self.next_checkpoint()
        # self.wind_var()
        self.state_divider()
        self.state_d=self.state
        # self.state_txt_display()
        self.flight_trace()
        self.max_alt = max(self.max_alt,self.vessel.flight(self.ref_frame_from).mean_altitude)
        # print(self.vessel.flight(self.ref_frame_from).latitude-self.lat_i,end="\r")
        # print(self.state[3], end="\r")

        # print("Pos: ({:.2f},{:.2f},{:.2f},{:.2f})".format(self.current_pos[0],self.current_pos[1],self.current_pos[2],self.current_pos[3]), end="\r")

    def record_video(self):
        print("Recording Started")
        fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
        bounding_box = {'top': 50, 'left': 65, 'width': 1570, 'height': 1030}

        sct = mss()

        frame_width = 1920
        frame_height = 1080
        frame_rate = 10.0
        out = cv2.VideoWriter(self.save_file_ext, fourcc2, frame_rate,(frame_width, frame_height))
        strt_time = time.time()
        while True :
            if self.stop_video:
                break
            if ((self.start_video) and (self.start_time is not None)):
                #print('Entered')
                time_duration = time.time() - strt_time
                if (time_duration >= (1/self.frame_rate_video)):
                    strt_time = time.time()
                    current_time = int((strt_time - self.start_time))
                    #print("Writing Frame")
                    sct_img = sct.grab(bounding_box)
                    img = np.array(sct_img)
                    img = cv2.resize(img,(frame_width,frame_height))
                    frame2 = img
                    frame2 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    out.write(frame2)
                    # cv2.imshow('screen', img)
                    # vout.write(self.frame)
                else:
                    time.sleep(0.0005)
            else:
                time.sleep(0.001)

        out.release()

    def run(self):
        # video_thread = threading.Thread(target=self.record_video)
        # video_thread.start()
        # self.target_trace()
        # self.init_states()
        while True:
            # self.get_obs()
            if self.stop_processing:
                break

        cv2.destroyAllWindows()
                               
        self.stop_video = True
        time.sleep(3)

class KSPRocketEnv(threading.Thread):
    def __init__(self):

        self.control = KeyboardCtrl()
        self.preprocessing = RocketTracking()
        self.RLagent = Agent(action_dim = ACTION_DIM)
        self.game_sav_dir = '/home/xpsucla/Spacecraft/Kerbal Space Program/ksp-linux-1.12.5/KSP_linux/saves/'
        self.game_sav_name = 'Python_test_13'
        self.tosave = 0
        if(self.tosave==1):
            self.preprocessing.conn.space_center.save(self.game_sav_name)
            print('Saved!')
        else:
            self.preprocessing.conn.space_center.load(self.game_sav_name)

        self.ref_frame = self.preprocessing.vessel.orbit.body.reference_frame
        self.cb = self.preprocessing.conn.space_center.bodies['Kerbin']
        self.vf = self.preprocessing.vessel.flight(self.ref_frame)
        # self.auto_p.target_pitch = 0
        self.vessel_control = self.preprocessing.vessel.control
        self.vessel_control.sas = True
        self.vessel_control.rcs = True
        self.vessel_control.gear = False
        self.vessel_control.brakes = False
        self.vessel_control.input_mode = self.vessel_control.input_mode.override

        self.takeoff = False
        self.exitcode = False
        self.duration = 1.0
        self.duration_rst = 10.0
        self.keypress_delay = 0.1
        self.start_exp = 0
        self.state = self.preprocessing.state
        self.stop_fpga = False
        self.action1 = '00'
        self.action2 = '00'
        self.action3 = '00'
        self.action4 = '00'
        self.human_actions = ['00','00','00','00']
        self.yaw = 0.0
        self.crashed = False
        self.oob = False
        self.crash_thres = -5
        self.crash_quit = False
        # self.reached_target = False
        self.target_quit = False
        self.curr_pos = list(self.preprocessing.current_pos)
        self.vxyz = self.preprocessing.vs
        self.vwind = self.vf.drag_coefficient
        self.dz = 0.0
        self.dz_gain = 10.
        self.dz_lim = [0.45,0.15]
        self.dy = 0.0
        self.dy_gain = [0.,0.]
        self.dx = 0.0
        self.dx_gain = 0.0
        self.dfb = 0.0
        self.dlr = 0.0
        self.dfb_gain = 2.#0.0#
        self.dlr_gain = 2.#0.0#
        self.dyaw = 0.0
        self.dyaw_gain = [0.,0.]

        self.pitch_i = self.vf.pitch
        self.pitch_thres = 10
        self.heading_i = self.vf.heading
        self.heading_thres = 10
        self.roll_i = self.vf.roll
        self.roll_thres = 10
        self.dep_par=0
        self.dep_brake=0
        self.brake_thres_vel = 150

        self.ch_check = False
        self.vw = self.preprocessing.vw
        self.force = self.preprocessing.f
        self.pitch_lim = 50
        self.heading_lim = 50

        self.episodes = NUM_EPS
        # self.episode = 1
        self.episode = epss
        self.states = self.preprocessing.state_all
        self.exp_time = 60
        self.landing_rocket_var = 0
        self.takeoff_rocket_var = 0
        self.preprocessing.z_i = self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).mean_altitude
        self.K_C = 0.0
        self.K_P1 = 20
        self.K_P2 = 20
        self.K_P3 = -2
        self.K_I = -0.0
        self.K_D = 0.0
        self.dz_gain_setalt = 20.0
        # self.vw = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

        # self.drag = self.vf.drag_coefficient

        super().__init__()
        super().start()

    def start(self):
        pass
   
    def takeoff_rocket_RL(self):
        if ((self.takeoff_rocket_var == 1) and (self.takeoff is False)):
            self.vessel_control.throttle = 0.0
            self.dz = 0.0
            self.vessel_control.sas = True
            self.vessel_control.rcs = True
            self.vessel_control.gear = True
            self.vessel_control.brakes = False
            self.vessel_control.input_mode = self.vessel_control.input_mode.override
            self.vessel_control.activate_next_stage()
            self.takeoff = True
            time.sleep(3)

    def land_rocket_RL(self):
        print("Landing")
        if (((self.landing_rocket_var == 1) and (self.takeoff is True)) or (self.control.quit())):
            self.vessel_control.throttle = 0.0
            self.vessel_control.brakes=False
            self.vessel_control.gear=False
            self.takeoff = False

    
    def crash_check(self):
        if(np.abs(self.preprocessing.pitch_angle)>self.pitch_lim or np.abs(self.preprocessing.heading_angle)>self.heading_lim or\
            (self.preprocessing.max_alt>85 and self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).mean_altitude < 80)):
            self.crashed = True
        if(np.any(np.abs(self.states)>100)):
            self.oob = True

    def check_done(self,dt):
        if(dt >= self.exp_time or self.oob == True or self.crashed == True or self.preprocessing.chp_num == self.preprocessing.total_chp):
            if(self.oob == True):
                print('Rocket out of bound!\n')
            if(self.crashed == True):
                print('Rocket crashed!\n')
            if(self.preprocessing.chp_num==self.preprocessing.total_chp):
                print('Experiment Complete!\n')
                # self.preprocessing.reset_chp()
                # self.preprocessing.mid = 1
            return 1
        else:
            return 0
        
    def P_cntrl(self):
        ts = time.time()
        while time.time()-ts < self.duration:
            vs = np.array(self.preprocessing.conn.space_center.transform_direction(self.preprocessing.vessel.velocity\
                                                                                   (self.preprocessing.ref_frame_from),\
                                                                                    self.preprocessing.ref_frame_from,\
                                                                                        self.preprocessing.ref_frame_to))
            # print(vs)
            vx1 = vs[1]
            vx2 = vs[2]
            vx3 = vs[0]
            proportion1 = vx1 - self.dfb
            proportion2 = vx2 - self.dlr
            proportion3 = vx3 - self.dz
            x_cntrl1 = self.K_P1 * proportion1
            x_cntrl2 = self.K_P2 * proportion2
            x_cntrl3 = self.K_P3 * proportion3
            x_cntrl1 = np.clip(x_cntrl1,-1.,1.)
            x_cntrl2 = np.clip(x_cntrl2,-1.,1.)
            x_cntrl3 = np.clip(x_cntrl3,0.15,0.75)
            self.vessel_control.up = x_cntrl1
            self.vessel_control.right = -x_cntrl2
            self.vessel_control.throttle = x_cntrl3

    def P_cntrl_rst(self):
        ts = time.time()
        while time.time()-ts < self.duration_rst:
            vs = np.array(self.preprocessing.conn.space_center.transform_direction(self.preprocessing.vessel.velocity\
                                                                                   (self.preprocessing.ref_frame_from),\
                                                                                    self.preprocessing.ref_frame_from,\
                                                                                        self.preprocessing.ref_frame_to))
            # print(vs)
            vx1 = vs[1]
            vx2 = vs[2]
            vx3 = vs[0]
            proportion1 = vx1 - self.dfb
            proportion2 = vx2 - self.dlr
            proportion3 = vx3 - self.dz
            x_cntrl1 = self.K_P1 * proportion1
            x_cntrl2 = self.K_P2 * proportion2
            x_cntrl3 = self.K_P3 * proportion3
            x_cntrl1 = np.clip(x_cntrl1,-1.,1.)
            x_cntrl2 = np.clip(x_cntrl2,-1.,1.)
            x_cntrl3 = np.clip(x_cntrl3,0.15,0.75)
            self.vessel_control.up = x_cntrl1
            self.vessel_control.right = -x_cntrl2
            self.vessel_control.throttle = x_cntrl3

    def P_cntrl_setalt(self):
        ts = time.time()
        while time.time()-ts < self.duration_rst:
            vs = np.array(self.preprocessing.conn.space_center.transform_direction(self.preprocessing.vessel.velocity\
                                                                                   (self.preprocessing.ref_frame_from),\
                                                                                    self.preprocessing.ref_frame_from,\
                                                                                        self.preprocessing.ref_frame_to))
            # print(vs)
            vx1 = vs[1]
            vx2 = vs[2]
            vx3 = vs[0]
            proportion1 = vx1 - self.dfb
            proportion2 = vx2 - self.dlr
            proportion3 = vx3 - self.dz_gain_setalt
            x_cntrl1 = self.K_P1 * proportion1
            x_cntrl2 = self.K_P2 * proportion2
            x_cntrl3 = self.K_P3 * proportion3
            x_cntrl1 = np.clip(x_cntrl1,-1.,1.)
            x_cntrl2 = np.clip(x_cntrl2,-1.,1.)
            x_cntrl3 = np.clip(x_cntrl3,0.15,0.5)
            self.vessel_control.up = x_cntrl1
            self.vessel_control.right = -x_cntrl2
            self.vessel_control.throttle = x_cntrl3

    def step(self):
        if(self.crashed is False):
            self.dfb = 0.0
            self.dlr = 0.0
            self.dz = 0.0
            # self.action = 2 if self.preprocessing.chp_num==0 else 3
            # ind_act = 0 if self.preprocessing.mid == 1 else 1

            if (self.action == 0):
                # self.dy = -self.dy_gain[ind_act]
                self.dfb = -self.dfb_gain
            elif (self.action == 1):
                # self.dy = self.dy_gain[ind_act]
                self.dfb = self.dfb_gain
                        
            if (self.action == 2):
                self.dz = self.dz_gain
            elif (self.action == 3):
                self.dz = -self.dz_gain

            if (self.action == 4):
                # self.dyaw = self.dyaw_gain[ind_act]
                self.dlr = self.dlr_gain
            elif (self.action == 5):
                # self.dyaw = -self.dyaw_gain[ind_act]
                self.dlr = -self.dlr_gain

            # print(ind_act)
            # self.dz = np.clip(self.dz,0,self.dz_lim[ind_act]) if ind_act==0 else np.clip(self.dz,self.dz_lim[ind_act],0.4)
            # self.vessel_control.throttle = self.dz
            self.P_cntrl()
            # time.sleep(self.keypress_delay)

    def train(self):#agent: Agent, env: AirSimDroneEnv, episodes: int):
        """Train `agent` in `env` for `episodes`
        Args:
        agent (Agent): Agent to train
        episodes (int): Number of episodes to train
        """
        input_arr = tf.random.uniform((1, num_inp))
        model = self.RLagent.policy_net
        outputs = model(input_arr)
        model._set_inputs(input_arr)
        
        while self.episode <= self.episodes:
            self.preprocessing.reset_chp()
            self.takeoff_rocket_var = 1
            self.takeoff_rocket_RL()
            self.preprocessing.lat_i = self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).latitude
            self.preprocessing.long_i = self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).longitude
            self.preprocessing.z_i = self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).mean_altitude
            self.preprocessing.get_alt()
            # self.preprocessing.init_states()
            # self.preprocessing.max_alt = self.preprocessing.z_i
            # print(self.preprocessing.current_pos)
            if(self.exitcode == True):
                break
            # if(ch_check == True):
            #     self.episode -= 1
            # model = get_model()
            if(self.episode == epss):
                model = self.RLagent.policy_net
                model.load_weights(mdir_epload)
                model.optimizer = self.RLagent.optimizer
                model.compile()
            else:
                model = self.RLagent.policy_net
                model.optimizer = self.RLagent.optimizer
                model.compile()
                model.save(mdir + str(self.episode),save_format='tf',include_optimizer=True)
            # model = self.RLagent.policy_net
            # model.optimizer = self.RLagent.optimizer
            # model.compile()
            # model.save(mdir + str(self.episode),save_format='tf',include_optimizer=True)

            done = False
            time.sleep(5.0)
            if(self.preprocessing.alt_r==-1):
                self.dfb=0.0
                self.dlr=0.0
                self.dz=0.0
                self.P_cntrl_setalt()
                self.P_cntrl_rst()
                self.preprocessing.lat_i = self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).latitude
                self.preprocessing.long_i = self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).longitude
                self.preprocessing.z_i = self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).mean_altitude

            self.preprocessing.init_states()
            self.preprocessing.get_obs()
            self.preprocessing.max_alt = self.preprocessing.z_i

            state_c = self.preprocessing.state_all
            print(f'Initial State: ({int(state_c[0]-state_c[1])},{int(state_c[2]-state_c[3])},{int(state_c[4]-state_c[5])})')
            self.states = self.preprocessing.state_all#np.array([100,100,100,100])
            total_reward = 0
            self.chp_flag = 0
            rewards = []
            states = []
            actions = []
            vxyzw = []
            time_array = []
            target_devs = []
            # crash = []
            time.sleep(0.2)
            # env.takeoff_drone_RL(1)
            ts = time.time()
            while not done:
                if (self.control.quit()):
                    print("Quitting Code")
                    self.exitcode = True
                    break
                # print(self.states)
                self.action = self.RLagent.get_action(self.states)
                # print(self.preprocessing.mid)
                self.step()
                self.preprocessing.get_obs()
                self.preprocessing.next_checkpoint()
                next_state = self.preprocessing.state_all#np.array([100,100,100,100])
                # ch_check = self.preprocessing.next_checkpoint(fac)
                self.crash_check()
                reward = reward_scheme(np.array(self.states))
                # print(time.time() - ts,'\n')
                done  = self.check_done(time.time() - ts)
                rewards.append(reward)
                states.append(list(self.states))
                actions.append(self.action)
                vxyzw.append(self.preprocessing.vw)
                target_devs.append(self.preprocessing.current_pos)
                # print(self.states)
                time_array.append(time.time() - ts)
                self.states = next_state
                total_reward += reward

                # if self.preprocessing.reached_checkpoint :
                #     done  = True
                #     self.preprocessing.reached_checkpoint = False
                
                # if (self.preprocessing.reached_target):
                #     print("Target Reached!")
                #     self.exitcode = True
                #     done = True
    
                if done:
                    self.crashed = False
                    self.oob = False
                    self.takeoff = False
                    
                    learn_start = time.time()

                    self.RLagent.learn(states, rewards, actions)

                    learning_time = time.time() - learn_start

                    print(f'Learning Time: {learning_time} s')
                    checkpoint_num = self.preprocessing.chp_num

                    with open(mdir_dat + str(self.episode) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump([states, rewards, actions,time_array,vxyzw,learning_time,target_devs,checkpoint_num], f)
                    print("\n")
                    print(f"Episode#:{self.episode} ep_reward:{np.mean(np.array(rewards))} CP: {self.preprocessing.chp_num}", end="\n")
            self.episode += 1
            print('States:',states[-10:])
            print('Actions:',actions[-10:])
            self.preprocessing.reset_chp()
            self.reset_rocket()
            # time.sleep(5)
        
        self.landing_rocket_var = 1
        self.land_rocket_RL()
        EXP_END = 1
        self.exitcode = True
    
    def reset_rocket(self):
        self.preprocessing.conn.space_center.load(self.game_sav_name)
        time.sleep(8)

    def run(self):
        print("Rocket is ready to launch")

        time.sleep(2)
        strt_time = time.time()
        frame_num = 0
        while True:
            time.sleep(0.1)
            if not self.exitcode:
                self.train()

            if self.exitcode:
                self.preprocessing.stop_processing = True
                time.sleep(2)
                break

        self.reset_rocket()

if __name__ == "__main__":
    ksp_rl = KSPRocketEnv()
    # Start RL
    ksp_rl.start()
    time.sleep(1)