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

# Rocket Build: FLT800,mk1 command pod,mk16 parachute, airbrakes, rv105 thruster, delta deluxe winglet,lvt30 fuel,lt2 landing strut
# Cheat menu right shift+F12

driving_agent = 'SNIC'
mdir = '/home/xpsucla/Spacecraft/Kerbal Space Program/Python files/Python Data/'
version = 7 #3,4,
date_dir = '/160124/'
heat_dir = 'Heat/'
save_dir = mdir + driving_agent + date_dir + heat_dir + 'v' + str(version)
home_dir = mdir + driving_agent + date_dir + heat_dir
fname_external = "External_Video_" + driving_agent + "_v" + str(version) + ".avi"

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
        # self.xtarget_temp = np.array([self.lat_i+0.02,self.lat_i+0.04,self.lat_i+0.05,self.lat_i+0.06,self.lat_i+0.075,self.lat_i+0.09,self.lat_i+0.10,self.lat_i+0.12,self.lat_i+0.13,self.lat_i+0.14])
        # self.ytarget_temp = np.array([self.long_i+0.0,self.long_i+0.005,self.long_i+0.01,self.long_i+0.02,self.long_i+0.035,self.long_i+0.05,self.long_i+0.065,self.long_i+0.08,self.long_i+0.10,self.long_i+0.13])
        # self.ztarget_temp = np.array([self.z_i+1000,self.z_i+2000,self.z_i+3500,self.z_i+5000,self.z_i+7000,self.z_i+9000,self.z_i+11000,self.z_i+14000,self.z_i+17000,self.z_i+20000])
        # self.xtarget_temp = np.array([self.lat_i+0.00,self.lat_i+0.0015,self.lat_i+0.003,self.lat_i+0.005,self.lat_i+0.008,self.lat_i+0.016,self.lat_i+0.019,self.lat_i+0.021,self.lat_i+0.0215,self.lat_i+0.022])
        # self.ytarget_temp = np.array([self.long_i-0.00,self.long_i-0.0015,self.long_i-0.003,self.long_i-0.005,self.long_i-0.008,self.long_i-0.016,self.long_i-0.019,self.long_i-0.021,self.long_i-0.0215,self.long_i-0.022])
        # # self.ytarget_temp = np.array([self.long_i-0.00,self.long_i-0.00,self.long_i-0.00,self.long_i-0.00,self.long_i-0.00,self.long_i-0.0,self.long_i-0.0,self.long_i-0.0,self.long_i-0.0,self.long_i-0.0])
        # self.ztarget_temp = np.array([self.z_i+1000,self.z_i+2500,self.z_i+4500,self.z_i+7000,self.z_i+10000,self.z_i+9500,self.z_i+6000,self.z_i+3000,self.z_i+500,self.z_i-75])
        # self.xtarget_temp = np.array([self.lat_i+0.08,self.lat_i+0.04])
        # self.ytarget_temp = np.array([self.long_i+0.0,self.long_i+0.04])
        # self.ztarget_temp = np.array([self.z_i+2000,self.z_i+5000])
        self.chp_div = 4000
        self.xtarget_temp = np.linspace(self.lat_i,self.lat_i+0.022,num=self.chp_div)
        self.ytarget_temp = np.linspace(self.long_i,self.long_i-0.022,num=self.chp_div)
        self.ztarget_temp = np.append(np.linspace(self.z_i+200,self.z_i+8000,num=self.chp_div//2),np.linspace(self.z_i+8000,self.z_i-75,num=self.chp_div//2))

        print('Latitude:',self.xtarget_temp[-1])
        print('Longitude:',self.ytarget_temp[-1])
        self.target_pos = np.array([self.xtarget_temp[-1], self.ytarget_temp[-1], self.ztarget_temp[-1],0.0])
        self.target_dev = self.target_pos - self.current_pos
        self.start_time = None
        self.ylim = -100
        self.reached_target = False
        self.state_gains = [.6,.6,0.3,0.1]
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
        self.reached_checkpoint = True
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
        self.chp_mid = self.total_chp/2
        # self.camera_vessel.mode = self.camera_vessel.mode.map
        self.s_d = [self.vessel.flight(self.ref_frame_from).latitude,self.vessel.flight(self.ref_frame_from).longitude,\
                           self.vessel.flight(self.ref_frame_from).mean_altitude]  

        self.show_states = False
        self.camera_vessel.heading = 0.0
        self.camera_vessel.pitch = 0.0
        # print(self.camera_vessel.pitch)
        self.camera_vessel.distance = 30.0
        self.vp = self.vessel.parts.controlling
        self.vw = np.zeros(3)
        self.f = np.zeros(3)
        self.kk = []
        self.vw_angle = 0.0
        self.pitch_angle = 0.0
        self.heading_angle = 0.0
        # self.chp_ts = time.time()
        self.chp_dt_sl_init = 1
        self.chp_dt = .07
        self.chp_dt_p = .2
        self.start_chp = 0
        self.start_inf = 0
        self.chp_mid_init = 5
        self.chp_mid_min = (self.total_chp/2)-55
        self.chp_mid_max = (self.total_chp/2)


        super().__init__()
        super().start()

    def start(self):
        pass
 

    def state_txt_display(self):
        if(time.time()-self.txt_time > self.dt_txt and self.show_states):
            if(self.start_txt !=0):
                self.txt.remove()
                self.ang_txt.remove()
            self.start_txt = 1
            # self.str_txt = '['+str(int(self.state[0]))+','+str(int(self.state[1]))+','+str(int(self.state[2]))+','+str(int(self.state[3]))+']'
            self.str_txt = '['+str(int(self.state[0]))+','+str(int(self.state[1]))+','+str(int(self.state[2]))+']'
            self.txt = self.conn.drawing.add_text(self.str_txt,self.vessel.surface_reference_frame,(-30,40,40),self.oi)
            self.txt.size = 100
            self.txt.style = self.txt.style.bold

            self.pitch_angle = self.vessel.flight(self.ref_frame_from).pitch - self.p_i
            self.heading_angle = self.vessel.flight(self.ref_frame_from).heading - self.h_i
            self.str_angtxt = '[\u03B1: '+str(int(self.pitch_angle*10)/10)+'\u00b0, \u03B2: '+str(int(self.heading_angle*10)/10)+'\u00b0]'
            self.ang_txt = self.conn.drawing.add_text(self.str_angtxt,self.vessel.surface_reference_frame,(-45,40,40),self.oi)
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
        s = [self.lat_i,self.long_i,self.z_i]
        tt_1 = self.vessel.orbit.body.position_at_altitude(s[0],s[1],s[2], self.ref_frame_from)
        tt_2 = self.vessel.orbit.body.position_at_altitude(self.xtarget_temp[0],self.ytarget_temp[0]\
                                                        ,self.ztarget_temp[0]+self.z_i, self.ref_frame_from)  
        tt_1 = np.array(tt_1)
        tt_2 = np.array(tt_2)
        ttt_1 = self.conn.drawing.add_line((tt_1[0],tt_1[1],tt_1[2]), (tt_2[0],tt_2[1],tt_2[2]), self.ref_frame_from)
        ttt_1.color = (250./255,12./255,246./255)
        ttt_1.thickness = 1

        for i in range(1,self.total_chp):
            tt_1 = self.vessel.orbit.body.position_at_altitude(self.xtarget_temp[i-1],self.ytarget_temp[i-1]\
                                                            ,self.ztarget_temp[i-1]+self.z_i, self.ref_frame_from)  
            tt_2 = self.vessel.orbit.body.position_at_altitude(self.xtarget_temp[i],self.ytarget_temp[i]\
                                                            ,self.ztarget_temp[i]+self.z_i, self.ref_frame_from)  
            tt_1 = np.array(tt_1)
            tt_2 = np.array(tt_2)
            ttt_1 = self.conn.drawing.add_line((tt_1[0],tt_1[1],tt_1[2]), (tt_2[0],tt_2[1],tt_2[2]), self.ref_frame_from)
            ttt_1.color = (250./255,12./255,246./255)
            ttt_1.thickness = 1


    def waypoint_tracker(self):
        if(self.chp_num>0):
            self.ll2.remove()
            self.ll3.remove()
        t_size = 0.0001
        self.poss_3 = self.vessel.orbit.body.position_at_altitude(self.xtarget_temp[self.chp_num]-t_size, self.ytarget_temp[self.chp_num]-t_size, self.ztarget_temp[self.chp_num]+self.z_i, self.ref_frame_from)
        self.poss_4 = self.vessel.orbit.body.position_at_altitude(self.xtarget_temp[self.chp_num]+t_size, self.ytarget_temp[self.chp_num]+t_size, self.ztarget_temp[self.chp_num]+self.z_i, self.ref_frame_from)
        self.poss_5 = self.vessel.orbit.body.position_at_altitude(self.xtarget_temp[self.chp_num]+t_size, self.ytarget_temp[self.chp_num]-t_size, self.ztarget_temp[self.chp_num]+self.z_i, self.ref_frame_from)
        self.poss_6 = self.vessel.orbit.body.position_at_altitude(self.xtarget_temp[self.chp_num]-t_size, self.ytarget_temp[self.chp_num]+t_size, self.ztarget_temp[self.chp_num]+self.z_i, self.ref_frame_from)
        self.poss_3 = np.array(self.poss_3)
        self.poss_4 = np.array(self.poss_4)
        self.poss_5 = np.array(self.poss_5)
        self.poss_6 = np.array(self.poss_6)
        self.ll2 = self.conn.drawing.add_line((self.poss_3[0],self.poss_3[1],self.poss_3[2]), (self.poss_4[0],self.poss_4[1],self.poss_4[2]), self.ref_frame_from)
        self.ll2.color = (250./255,12./255,246./255)
        self.ll2.thickness = 3
        self.ll3 = self.conn.drawing.add_line((self.poss_5[0],self.poss_5[1],self.poss_5[2]), (self.poss_6[0],self.poss_6[1],self.poss_6[2]), self.ref_frame_from)
        self.ll3.color = (250./255,12./255,246./255)
        self.ll3.thickness = 3

    def wind_var(self):
        rmin=15000
        rmax=25000
        
        if(self.chp_num!=0 and self.chp_num%100==0 and time.time()-self.wind_ts>2):
            vx = np.random.choice(np.append(np.arange(-rmax,-rmin),np.arange(rmin,rmax)))/1000.0
            vy = np.random.choice(np.append(np.arange(-rmax,-rmin),np.arange(rmin,rmax)))/1000.0
            vz = np.random.choice(np.append(np.arange(-rmax,-rmin),np.arange(rmin,rmax)))/1000.0
            self.vw = np.array([vx,0,vz])
            print('Wind Speed:',self.vw)
            self.wind_ts = time.time()

        comrf = self.vp.center_of_mass_reference_frame
        com = self.vp.center_of_mass(comrf)
        vesvel = np.abs(np.array(self.vf.velocity))
        if(self.chp_num<200):
            k = np.abs(np.array(self.vf.aerodynamic_force)/(vesvel**2))
            # k = np.min(k)
            # k = np.array(self.vf.drag_coefficient)+np.array(self.vf.lift_coefficient)
            self.kk.append(k)
            # print(np.shape(np.array(self.kk)))
        else:
            k = np.mean(np.array(self.kk),axis=0)
        # vesvel = np.array([vesvel[0],0,vesvel[2]])
        forcee = k*self.vw**2 + 2*k*vesvel*self.vw
        self.f = forcee
        if(self.chp_num>=150 and ~np.isnan(forcee[0]) and np.abs(vesvel[2])>2):
            self.vp.instantaneous_force(forcee,com,comrf)
            # print('Force Applied!')
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
            self.wind_txt = self.conn.drawing.add_text(self.str_wtxt,self.vessel.surface_reference_frame,(-15,40,40),self.oi)
            self.wind_txt.size = 100
            self.wind_txt.style = self.wind_txt.style.bold
            self.wind_txt_time = time.time()

    def change_chp(self):
        if(self.chp_num < self.total_chp-1):
            if(self.chp_num <= self.chp_mid_init and (time.time()-self.chp_ts>self.chp_dt_sl_init ) and self.start_chp==1):
                self.chp_num+=1
                self.waypoint_tracker()
                self.chp_ts = time.time()
            elif(self.chp_num > self.chp_mid_init and (self.chp_num <= self.chp_mid_min or self.chp_num >= self.chp_mid_max) and (time.time()-self.chp_ts>self.chp_dt ) and self.start_chp==1):#or abs(self.state[2]) < 15
                self.chp_num+=1
                self.waypoint_tracker()
                self.chp_ts = time.time()
                # print(self.chp_num)
            elif((self.chp_num > self.chp_mid_min and self.chp_num < self.chp_mid_max) and self.start_chp==1 and (time.time()-self.chp_ts>self.chp_dt_p)):
                self.chp_num+=1
                self.waypoint_tracker()
                self.chp_ts = time.time()

    def get_obs(self):
        self.change_chp()
        self.current_pos[3] = self.vessel.flight(self.ref_frame_from).roll - self.roll_i
        pos_ar = self.vessel.velocity(self.ref_frame_from)
        self.vs = np.array(self.conn.space_center.transform_direction(pos_ar,self.ref_frame_from,self.ref_frame_to))
        self.ts = time.time()
        self.current_pos[0]= (math.pi/180) * self.planet_radius * (self.vessel.flight(self.ref_frame_from).latitude - self.lat_i)
        self.current_pos[1]= (math.pi/180) * self.planet_radius * (self.vessel.flight(self.ref_frame_from).longitude - self.long_i)
        self.current_pos[2]= self.vessel.flight(self.ref_frame_from).mean_altitude - self.z_i
        self.state[0] = -np.clip((self.current_pos[0]-((math.pi/180) * self.planet_radius * (self.xtarget_temp[self.chp_num] - self.lat_i)))*self.state_gains[0],-100,100)
        self.state[1] = np.clip((self.current_pos[1]-((math.pi/180) * self.planet_radius * (self.ytarget_temp[self.chp_num] - self.long_i)))*self.state_gains[1],-100,100)
        self.state[2] = np.clip((self.current_pos[2]-self.ztarget_temp[self.chp_num])*self.state_gains[2],-100,100)
        self.state[3] = np.clip(self.current_pos[3]*self.state_gains[3],-100,100)
        self.target_dev = self.target_pos - self.current_pos
        self.wind_var()
        self.state_d=self.state
        self.state_txt_display()
        self.flight_trace()
        self.max_alt = max(self.max_alt,self.vessel.flight(self.ref_frame_from).mean_altitude)
        # print(self.vessel.flight(self.ref_frame_from).latitude-self.lat_i,end="\r")
        # print(self.state[3], end="\r")

        # print("Pos: ({:.2f},{:.2f},{:.2f},{:.2f})".format(self.current_pos[0],self.current_pos[1],self.current_pos[2],self.current_pos[3]), end="\r")

    def record_video(self):
        print("Recording Started")
        fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
        # bounding_box = {'top': 50, 'left': 65, 'width': 1570, 'height': 1030}
        bounding_box = {'top': 0, 'left': 0, 'width': 1620, 'height': 1080}

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
        self.target_trace()
        self.waypoint_tracker()
        video_thread = threading.Thread(target=self.record_video)
        video_thread.start()
        self.chp_ts = time.time()
        self.wind_ts = time.time()
        while True:
            self.get_obs()
            if self.stop_processing:
                break

        cv2.destroyAllWindows()
                               
        self.stop_video = True
        time.sleep(3)

class KSPRocketEnv(threading.Thread):
    def __init__(self):

        self.control = KeyboardCtrl()
        self.preprocessing = RocketTracking()
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
        self.keypress_delay = 0.008
        self.duration = 0.1
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
        self.crash_thres = -5
        self.crash_quit = False
        # self.reached_target = False
        self.target_quit = False
        self.curr_pos = list(self.preprocessing.current_pos)
        self.vxyz = self.preprocessing.vs
        self.vwind = self.vf.drag_coefficient

        # self.dz = 0.0
        # self.dz_gain = [0.1,0.025]
        # self.dz_lim = [0.45,0.15]
        # self.dy = 0.0
        # self.dy_gain = [0.1,0.1]
        # self.dx = 0.0
        # self.dx_gain = 0.05
        # self.dfb = 0.0
        # self.dlr = 0.0
        # self.dfb_gain = [.8,.8]#0.0#
        # self.dlr_gain = [.8,.8]#0.0#
        # self.dyaw = 0.0
        # self.dyaw_gain = [0.1,0.1]

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

        self.K_C = 0.0
        self.K_P1 = 20
        self.K_P2 = 20
        self.K_P3 = -2
        self.K_I = -0.0
        self.K_D = 0.0

        self.dz = 0.0
        self.dz_gain = 50
        self.dz_lim = [0.45,0.15]
        self.dy = 0.0
        self.dy_gain = 0.1
        self.dx = 0.0
        self.dx_gain = 0.0
        self.dfb = 0.0
        self.dlr = 0.0
        self.dfb_gain = 5#0.0#
        self.dlr_gain = 5#0.0#
        self.dyaw = 0.0
        self.dyaw_gain = 0.1
        # self.vw = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

        # self.drag = self.vf.drag_coefficient

        super().__init__()
        super().start()

    def start(self):
        pass
    
    def takeoff_rocket(self):
        if ((self.control.takeoff() == '1') and (self.takeoff is False)):
            self.vessel_control.throttle = 0.0
            self.vessel_control.activate_next_stage()
            self.preprocessing.z_i = self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).mean_altitude
            self.takeoff = True
    

    def land_rocket(self):   
        if (((self.control.landing() == '1') and (self.takeoff is True)) or (self.control.quit())):
            self.takeoff = False
            # self.preprocessing.conn.space_center.load(self.game_sav_name)
        elif (self.crashed and self.takeoff is True):
            # self.preprocessing.conn.space_center.load(self.game_sav_name)
            self.takeoff = False
            self.crash_quit = True
            self.preprocessing.start_video = False
        elif (self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).mean_altitude <= self.preprocessing.z_i+30 and self.takeoff and self.dep_par==1):
            # self.preprocessing.conn.space_center.load(self.game_sav_name)
            print('Reached target!')
            time.sleep(2)
            self.vessel_control.throttle = 0.0
            time.sleep(10)
            self.target_quit = True
            self.preprocessing.start_video = False
        elif(self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).mean_altitude <= self.preprocessing.z_i+500 and self.takeoff and self.dep_par == 0 and self.dep_brake==1):
            print('Deploying parachute and landing gear!')
            self.vessel_control.brakes=True
            self.vessel_control.gear=True
            self.vessel_control.activate_next_stage()
            self.dep_par=1
        elif(self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).mean_altitude >= self.preprocessing.z_i+8000 and self.takeoff and abs(self.preprocessing.vessel.velocity(self.preprocessing.ref_frame_from)[-1]) < self.brake_thres_vel and self.dep_brake==0):
            print('Deploying air brakes!')
            self.vessel_control.brakes=True
            self.dep_brake = 1

    
    def recording_state(self):
        if ((self.control.start_experiment() == '1') and (self.start_exp == 0)):
            time.sleep(1.0)
            print('Experiment Started')
            self.start_exp = 1
            self.preprocessing.start_chp = 1
            self.preprocessing.start_video = True
            self.preprocessing.start_time = time.time()
            self.preprocessing.camera_vessel.distance = 200.0
            self.preprocessing.show_states = True

        elif ((self.control.end_experiment() == '1') and (self.start_exp == 1)):
            print("Experiment Ended")
            self.start_exp = 0
            self.preprocessing.start_video = False
        elif (self.preprocessing.vessel.flight(self.preprocessing.ref_frame_from).mean_altitude <= self.preprocessing.z_i-2 and (self.start_exp == 1)):
            print("Experiment Ended")
            self.start_exp = 0
        elif(self.crashed and self.start_exp == 1):
            print('Crashed!')
            print("Experiment Ended")
            self.start_exp = 0
        
                       
    def human_action_update(self):
        human_action1 = self.control.pitch()
        human_action2 = self.control.yaw()
        human_action3 = self.control.thrttle()
        human_action4 = self.control.roll()
        self.human_actions = [human_action1,human_action2,human_action3,human_action4]

    def action_update(self,act1,act2,act3,act4):
        self.action1 = act1
        self.action2 = act2
        self.action3 = act3
        self.action4 = act4
    
    def crash_check(self):
        # if(np.abs(self.preprocessing.pitch_angle)>self.pitch_lim or np.abs(self.preprocessing.heading_angle)>self.heading_lim\
        #    or np.any(np.abs(self.preprocessing.state)>=100)):
        if(np.any(np.abs(self.preprocessing.state)>=100)):
            self.crashed = True

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

    def rocket_control(self):
        # print(self.crashed)
        if(self.crashed is False):
            # self.dfb = 0.0
            # self.dlr = 0.0
            # self.dz = 0.0
            self.dy = 0.0
            self.dyaw = 0.0
            # ind_act = 0 if self.preprocessing.mid == 1 else 1

            if (self.control.pitch() == '01'):
                self.dy = -self.dy_gain
                self.dfb = -self.dfb_gain
            elif (self.control.pitch() == '10'):
                self.dy = self.dy_gain
                self.dfb = self.dfb_gain
            else:
                if self.start_exp==1:
                    if (self.action1 == '01'):
                        self.dy = self.dy_gain
                        self.dfb = self.dfb_gain
                    elif (self.action1 == '10'):
                        self.dy = -self.dy_gain
                        self.dfb = -self.dfb_gain
                        
            if (self.control.thrttle() == '01'):
                self.dz = self.dz_gain
            elif (self.control.thrttle() == '10'):
                self.dz = -self.dz_gain
            else:
                if self.start_exp==1:
                    if (self.action3 == '01'):
                        self.dz = self.dz_gain
                    elif (self.action3 == '10'):
                        self.dz = -self.dz_gain

            if (self.control.yaw() == '01'):
                self.dyaw = self.dyaw_gain
                self.dlr = self.dlr_gain
            elif (self.control.yaw() == '10'):
                self.dyaw = -self.dyaw_gain
                self.dlr = -self.dlr_gain
            else:
                if self.start_exp==1:
                    if (self.action2 == '01'):
                        self.dyaw = self.dyaw_gain
                        self.dlr = self.dlr_gain
                    elif (self.action2 == '10'):
                        self.dyaw = -self.dyaw_gain
                        self.dlr = -self.dlr_gain

            # print(ind_act)
            # self.dz = np.clip(self.dz,0,self.dz_lim[ind_act]) if ind_act==0 else np.clip(self.dz,self.dz_lim[ind_act],0.4)
            # self.vessel_control.throttle = self.dz
            self.P_cntrl()
            # self.vessel_control.pitch = self.dy
            # self.vessel_control.yaw = self.dyaw
            # time.sleep(self.keypress_delay)

   
    def update_state(self):
        #self.preprocessing.get_obs()
        self.state = self.preprocessing.state
        self.curr_pos = list(self.preprocessing.current_pos)
        self.vxyz = list(self.preprocessing.vs)
        self.vwind = list(self.preprocessing.vw)
        self.force = list(self.preprocessing.f)

    
    def reset_rocket(self):
        self.preprocessing.conn.space_center.load(self.game_sav_name)
        time.sleep(2)

    def run(self):
        print("Rocket is ready to launch")

        time.sleep(2)
        strt_time = time.time()
        frame_num = 0
        while True:
            self.crash_check()
            self.recording_state()
            if not self.takeoff:
                self.takeoff_rocket()
            else:
                self.land_rocket()
                self.rocket_control()
            
            # self.ch_check = self.preprocessing.next_checkpoint()
            time.sleep(0.01)
            
            if(self.control.quit() or self.crash_quit or self.target_quit) :
                print("Quitting Code")
                self.exitcode = True

            if self.exitcode:
                self.preprocessing.stop_processing = True
                time.sleep(2)
                self.stop_fpga = True
                # time.sleep(5)
                break

        self.reset_rocket()


class FPGAComm():

    def __init__(self):
        self.mainfunc = KSPRocketEnv()
        self.state = self.mainfunc.state
        self.start_exp = self.mainfunc.start_exp
        self.check_ch = 0
        self.mainfunc.start()
        self.fname = save_dir + '.pkl' #"FPGA_data_test_v35.pkl"
        self.save_data = True
        time.sleep(2)

    def find_device(self):
        """
        Find FX3 device and the corresponding endpoints (bulk in/out).
        If find device and not find endpoints, this may because no images are programed, we will program image;
        If image is programmed and still not find endpoints, raise error;
        If not find device, raise error.

        :return: usb device, usb endpoint bulk in, usb endpoint bulk out
        """

        # find device
        dev = usb.core.find(idVendor=0x04b4)
        intf = dev.get_active_configuration()[(0, 0)]

        # find endpoint bulk in
        ep_in = usb.util.find_descriptor(intf,
                                        custom_match=lambda e:
                                        usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN)

        # find endpoint bulk out
        ep_out = usb.util.find_descriptor(intf,
                                        custom_match=lambda e:
                                        usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_OUT)

        if ep_in is None and ep_out is None:
            print('Error: Cannot find endpoints after programming image.')
            return -1
        else:
            return dev, ep_in, ep_out
    
    def run(self):
        print("Start!")
        # find device
        usb_dev, usb_ep_in, usb_ep_out = self.find_device()
        usb_dev.set_configuration()
        
        # initial reset usb and fpga
        usb_dev.reset()
        if driving_agent=='SNIC':
            is_SNIC = True
        else:
            is_SNIC = False

        fpga_data_array = []
        time_array = []
        curr_pos_array = []
        vxyz_array = []
        vwind_array = []
        human_actions_array = []
        force_array=[]


        num = 64 * 1
        str_time = time.time()
        while not self.mainfunc.stop_fpga:
            self.mainfunc.update_state()
            self.state = self.mainfunc.state
            self.start_exp = self.mainfunc.start_exp
            #self.check_ch = int(self.mainfunc.ch_check)
            np_data1 = np.array([self.state[0],self.state[1],self.state[2], 0,self.start_exp],dtype=np.uint8)
            # np_data1 = np.array([0,0,-100,0,self.start_exp],dtype=np.uint8)
            np_data2 = np.random.randint(0, high=255, size = num-5, dtype=np.uint8)
            np_data = np.concatenate((np_data1,np_data2))
            wr_data = list(np_data)
            length = len(wr_data)
        
            # write data to ddr
            opu_dma(wr_data, num, 10, 0, usb_dev, usb_ep_out, usb_ep_in)
        
            # start calculation
            opu_run([], 0, 0, 3, usb_dev, usb_ep_out, usb_ep_in)

            # read data from FPGA
            rd_data = []
            opu_dma(rd_data, num, 11, 2, usb_dev, usb_ep_out, usb_ep_in)

            if is_SNIC:
                action1 = '{0:02b}'.format(int(rd_data[0]))
                action2 = '{0:02b}'.format(int(rd_data[1]))
                action3 = '{0:02b}'.format(int(rd_data[2]))
                action4 = '{0:02b}'.format(int(rd_data[3]))
                self.mainfunc.action_update(action1,action2,action3,action4)
            else:
                self.mainfunc.human_action_update()

            '''action3 = rd_data[0]
            action2 = rd_data[1]
            action1 = rd_data[2]'''

            if self.start_exp==1:
                fpga_data_array.append(rd_data)
                time_array.append((time.time()-str_time))
                curr_pos_array.append(self.mainfunc.curr_pos)
                vxyz_array.append(self.mainfunc.vxyz)
                vwind_array.append(self.mainfunc.vwind)
                human_actions_array.append(self.mainfunc.human_actions)
                force_array.append(self.mainfunc.force)
        
        if self.save_data:
            with open(self.fname, "wb") as fout:
                # default protocol is zero
                # -1 gives highest prototcol and smallest data file size
                pickle.dump((fpga_data_array, time_array, curr_pos_array, vxyz_array,vwind_array,human_actions_array,force_array), fout, protocol=-1)

if __name__ == "__main__":
    fpga_comm = FPGAComm()
    # Start the fpga communication
    fpga_comm.run()
    time.sleep(1)
    