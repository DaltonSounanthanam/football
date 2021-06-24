# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from matplotlib import pyplot as plt
import glob
import cv2
import os
import numpy as np
from datetime import datetime


### path to data files ###

path = ""

### load data files ###

game_data = pd.read_csv(path + "game_data.csv")

player_punt_data = pd.read_csv(path + "player_punt_data.csv")

play_information = pd.read_csv(path + "play_information.csv")

play_player_role_data = pd.read_csv(path + "play_player_role_data.csv")


### load NGS files ###

NGS_2016_pre = pd.read_csv('NGS-2016-pre.csv')

'''
NGS_2016_post = pd.read_csv('NGS-2016-post.csv')
NGS_2016_wk_1_6 = pd.read_csv('NGS-2016-reg-wk1-6.csv')
NGS_2016_wk_7_12 = pd.read_csv('NGS-2016-reg-wk7-12.csv')
NGS_2016_wk_13_17 = pd.read_csv('NGS-2016-reg-wk13-17.csv')

NGS_2017_pre = pd.read_csv('NGS-2017-pre.csv')
NGS_2017_post = pd.read_csv('NGS-2017-post.csv')
NGS_2017_wk_1_6 = pd.read_csv('NGS-2017-reg-wk1-6.csv')
NGS_2017_wk_7_12 = pd.read_csv('NGS-2017-reg-wk7-12.csv')
NGS_2017_wk_13_17 = pd.read_csv('NGS-2017-reg-wk13-17.csv')
'''
PuntCoverage_role = ['GL', 'GR', 'PLT', 'PLW', 'PLG', 'PLS', 'PRG', 'PRT', 'PRW', 'PC', 'PPR', 'P']
PuntReturn_role = ['VRo', 'VRi', 'PDR1', 'PDR2', 'PDR3', 'PDL1', 'PDL2', 'PDL3', 'VL', 'PFB', 'PR']

### building a play object ###

class play(object):
    
    def __init__(self, gamekey, playid, raw_play_data):
        prole = play_player_role_data[(play_player_role_data['GameKey'] == gamekey) & (play_player_role_data['PlayID'] == playid)]
        self.player_role = prole
        self.return_player_id = prole[prole['Role'].isin(PuntReturn_role)]
        self.coverage_player_id = prole[prole['Role'].isin(PuntCoverage_role)]
        
        play_data = raw_play_data[ (raw_play_data['GameKey'] == gamekey) & (raw_play_data['PlayID'] == playid)].sort_values(by = 'Time', ascending = True)
        play_data = play_data[play_data['GSISID'].isin(prole['GSISID'].tolist())]
        self.play_data = play_data
        self.P_ID = prole[prole['Role'] == 'P']['GSISID'].iloc[0]
        self.PR_ID = prole[prole['Role'] == 'PR']['GSISID'].iloc[0]
        self.timestamp = play_data[~play_data['Time'].duplicated(keep = 'first')]['Time'].tolist()
        
        self.starting = play_data[play_data['Time'] == self.timestamp[0]]
        sequences = []
        for t in self.timestamp:
            sequences.append(play_data[play_data['Time'] == t ])
        self.sequences = sequences
        
    
    def compute_speed(self, no = 1):
        frame_1 = self.sequences[no]
        XY_c = frame_1[frame_1['GSISID'].isin(self.coverage_player_id['GSISID'].tolist())]
        XY_r = frame_1[frame_1['GSISID'].isin(self.return_player_id['GSISID'].tolist())]
        time_1 = frame_1['Time'].iloc[0]
        datetime1 = datetime.strptime(time_1, '%Y-%m-%d %H:%M:%S.%f')
            
        frame_2 = self.sequences[abs(no - 1)]
        time_2 = frame_2['Time'].iloc[0]
        datetime2 = datetime.strptime(time_2, '%Y-%m-%d %H:%M:%S.%f')
            
        delta_t = (datetime1 - datetime2).microseconds
        if datetime1 < datetime2:
            delta_t = (datetime2 - datetime1).microseconds
        
        if delta_t <= 0:
            raise ValueError
        delta_t = delta_t/1000000
        
        speed_c = XY_c['dis']/delta_t
        speed_r = XY_r['dis']/delta_t
            
        return speed_c, speed_r
        
    def compute_speed2(self, frame_1, frame_2):
        player_list2 = frame_2['GSISID'].values.tolist()
        player_list1 = frame_1['GSISID'].values.tolist()
        
        time_1 = frame_1['Time'].iloc[0]
        datetime1 = datetime.strptime(time_1, '%Y-%m-%d %H:%M:%S.%f')
        
        time_2 = frame_2['Time'].iloc[0]
        datetime2 = datetime.strptime(time_2, '%Y-%m-%d %H:%M:%S.%f')
        
        delta_t = (datetime2 - datetime1).microseconds
        if delta_t <= 0:
            raise ValueError
        delta_t = delta_t/1000000
            
        speed = []
        for player2 in player_list2:
            if player2 in player_list1:
                XY_2 = frame_2[frame_2['GSISID'] == player2] 
                XY_1 = frame_1[frame_1['GSISID'] == player2]
                X_2 = XY_2['x']
                Y_2 = XY_2['y']
                X_1 = XY_1['x']
                Y_1 = XY_1['y']
                delta_X = X_2 - X_1
                delta_Y = Y_2 - Y_1
                sp = np.sqrt(delta_X**2 + delta_Y**2)/delta_t
                speed.append(sp)
            else:
                speed.append(1)
        return speed
        
   
            
    def plot_starting(self):
        XY_c = self.starting[self.starting['GSISID'].isin(self.coverage_player_id['GSISID'].tolist())]
        XY_r = self.starting[self.starting['GSISID'].isin(self.return_player_id['GSISID'].tolist())]
        XY_p = self.starting[self.starting['GSISID'] == self.P_ID]
        XY_pr = self.starting[self.starting['GSISID'] == self.PR_ID]
        X_c = XY_c['x'] - 10
        Y_c = XY_c['y']
        X_r = XY_r['x'] - 10
        Y_r = XY_r['y']
        X_p = XY_p['x'] - 10
        Y_p = XY_p['y']
        X_pr = XY_pr['x'] - 10
        Y_pr = XY_pr['y']
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim(-10, 110)
        plt.ylim(0, 53.3)
        plt.scatter(X_c, Y_c, c = 'r')
        plt.scatter(X_r, Y_r, c = 'b')
        plt.scatter(X_p, Y_p, c = 'orange')
        plt.scatter(X_pr, Y_pr , c = 'c')
        plt.axvline(x = 0, c = 'w', linewidth = 1)
        plt.axvline(x = 100, c = 'w', linewidth = 1)
        plt.axvline(x = 10, c = 'w', linewidth = 1)
        plt.axvline(x = 20, c = 'w', linewidth = 1)
        plt.axvline(x = 30, c = 'w', linewidth = 1)
        plt.axvline(x = 40, c = 'w', linewidth = 1)
        plt.axvline(x = 50, c = 'w', linewidth = 1)
        plt.axvline(x = 60, c = 'w', linewidth = 1)
        plt.axvline(x = 70, c = 'w', linewidth = 1)
        plt.axvline(x = 80, c = 'w', linewidth = 1)
        plt.axvline(x = 90, c = 'w', linewidth = 1)
        plt.axvline(x = 5, c = 'w', linewidth = 0.5)
        plt.axvline(x = 15, c = 'w', linewidth = 0.5)
        plt.axvline(x = 25, c = 'w', linewidth = 0.5)
        plt.axvline(x = 35, c = 'w', linewidth = 0.5)
        plt.axvline(x = 45, c = 'w', linewidth = 0.5)
        plt.axvline(x = 55, c = 'w', linewidth = 0.5)
        plt.axvline(x = 65, c = 'w', linewidth = 0.5)
        plt.axvline(x = 75, c = 'w', linewidth = 0.5)
        plt.axvline(x = 85, c = 'w', linewidth = 0.5)
        plt.axvline(x = 95, c = 'w', linewidth = 0.5)
        #ax.set_facecolor('#3f9b0b')
        ax.set_facecolor('#008000')
        
    def plot_starting_vector(self):
        XY_c = self.starting[self.starting['GSISID'].isin(self.coverage_player_id['GSISID'].tolist())]
        XY_r = self.starting[self.starting['GSISID'].isin(self.return_player_id['GSISID'].tolist())]
        XY_p = self.starting[self.starting['GSISID'] == self.P_ID]
        XY_pr = self.starting[self.starting['GSISID'] == self.PR_ID]
        
        X_p = XY_p['x'] - 10
        Y_p = XY_p['y']
        X_pr = XY_pr['x'] - 10
        Y_pr = XY_pr['y']
        
        X_c = XY_c['x'] - 10
        Y_c = XY_c['y']
        O_c = (XY_c['o'] + 90)*2*np.pi/360
        D_c = (XY_c['dir'] + 90)*2*np.pi/360
        U_c = np.cos(O_c) 
        V_c = np.sin(O_c)
        M_c = np.cos(D_c)
        N_c = np.sin(D_c)
        
        X_r = XY_r['x'] - 10
        Y_r = XY_r['y']
        O_r = (XY_r['o'] + 90)*2*np.pi/360
        D_r = (XY_r['dir'] + 90)*2*np.pi/360
        U_r = np.cos(O_r) 
        V_r = np.sin(O_r)
        M_r = np.cos(D_r)
        N_r = np.sin(D_r)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim(-10, 110)
        plt.ylim(0, 53.3)
        plt.scatter(X_c, Y_c, c = 'r')
        plt.scatter(X_r, Y_r, c = 'b')
        plt.scatter(X_p, Y_p, c = 'orange')
        plt.scatter(X_pr, Y_pr , c = 'c')
        plt.axvline(x = 0, c = 'w', linewidth = 1)
        plt.axvline(x = 100, c = 'w', linewidth = 1)
        plt.axvline(x = 10, c = 'w', linewidth = 1)
        plt.axvline(x = 20, c = 'w', linewidth = 1)
        plt.axvline(x = 30, c = 'w', linewidth = 1)
        plt.axvline(x = 40, c = 'w', linewidth = 1)
        plt.axvline(x = 50, c = 'w', linewidth = 1)
        plt.axvline(x = 60, c = 'w', linewidth = 1)
        plt.axvline(x = 70, c = 'w', linewidth = 1)
        plt.axvline(x = 80, c = 'w', linewidth = 1)
        plt.axvline(x = 90, c = 'w', linewidth = 1)
        plt.axvline(x = 5, c = 'w', linewidth = 0.5)
        plt.axvline(x = 15, c = 'w', linewidth = 0.5)
        plt.axvline(x = 25, c = 'w', linewidth = 0.5)
        plt.axvline(x = 35, c = 'w', linewidth = 0.5)
        plt.axvline(x = 45, c = 'w', linewidth = 0.5)
        plt.axvline(x = 55, c = 'w', linewidth = 0.5)
        plt.axvline(x = 65, c = 'w', linewidth = 0.5)
        plt.axvline(x = 75, c = 'w', linewidth = 0.5)
        plt.axvline(x = 85, c = 'w', linewidth = 0.5)
        plt.axvline(x = 95, c = 'w', linewidth = 0.5)
        ax.set_facecolor('#3f9b0b')
        
        speed_c, speed_r = self.compute_speed(0)
        
        ax.quiver( X_c, Y_c, U_c, V_c  , color = 'y', headwidth = 1, headlength  = 3, headaxislength = 3 )
        ax.quiver( X_r, Y_r, U_r, V_r, color = 'black', headwidth = 1, headlength  = 3, headaxislength = 3   )
        ax.quiver( X_r, Y_r, M_r, N_r, color = 'y', headwidth = 1, headlength  = 3, headaxislength = 3   )
        ax.quiver( X_c, Y_c, M_c, N_c, color = 'b', headwidth = 1, headlength  = 3, headaxislength = 3   )
        
    def plot_frame(self, no = 0):
        frame = self.sequences[no]
        XY_c = frame[frame['GSISID'].isin(self.coverage_player_id['GSISID'].tolist())]
        XY_r = frame[frame['GSISID'].isin(self.return_player_id['GSISID'].tolist())]
        XY_p = frame[frame['GSISID'] == self.P_ID]
        XY_pr = frame[frame['GSISID'] == self.PR_ID]
        X_c = XY_c['x'] - 10
        Y_c = XY_c['y']
        X_r = XY_r['x'] - 10
        Y_r = XY_r['y']
        X_p = XY_p['x'] - 10
        Y_p = XY_p['y']
        X_pr = XY_pr['x'] - 10
        Y_pr = XY_pr['y']
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim(-10, 110)
        plt.ylim(0, 53.3)
        plt.scatter(X_c, Y_c, c = 'r')
        plt.scatter(X_r, Y_r, c = 'b')
        plt.scatter(X_p, Y_p, c = 'orange')
        plt.scatter(X_pr, Y_pr , c = 'c')
        plt.axvline(x = 0, c = 'w', linewidth = 1)
        plt.axvline(x = 100, c = 'w', linewidth = 1)
        plt.axvline(x = 10, c = 'w', linewidth = 1)
        plt.axvline(x = 20, c = 'w', linewidth = 1)
        plt.axvline(x = 30, c = 'w', linewidth = 1)
        plt.axvline(x = 40, c = 'w', linewidth = 1)
        plt.axvline(x = 50, c = 'w', linewidth = 1)
        plt.axvline(x = 60, c = 'w', linewidth = 1)
        plt.axvline(x = 70, c = 'w', linewidth = 1)
        plt.axvline(x = 80, c = 'w', linewidth = 1)
        plt.axvline(x = 90, c = 'w', linewidth = 1)
        plt.axvline(x = 5, c = 'w', linewidth = 0.5)
        plt.axvline(x = 15, c = 'w', linewidth = 0.5)
        plt.axvline(x = 25, c = 'w', linewidth = 0.5)
        plt.axvline(x = 35, c = 'w', linewidth = 0.5)
        plt.axvline(x = 45, c = 'w', linewidth = 0.5)
        plt.axvline(x = 55, c = 'w', linewidth = 0.5)
        plt.axvline(x = 65, c = 'w', linewidth = 0.5)
        plt.axvline(x = 75, c = 'w', linewidth = 0.5)
        plt.axvline(x = 85, c = 'w', linewidth = 0.5)
        plt.axvline(x = 95, c = 'w', linewidth = 0.5)
        ax.set_facecolor('#3f9b0b')
        
    def plot_frame_vector(self, no = 0):
        frame = self.sequences[no]
        XY_c = frame[frame['GSISID'].isin(self.coverage_player_id['GSISID'].tolist())]
        XY_r = frame[frame['GSISID'].isin(self.return_player_id['GSISID'].tolist())]
        XY_p = frame[frame['GSISID'] == self.P_ID]
        XY_pr = frame[frame['GSISID'] == self.PR_ID]
       
        X_c = XY_c['x'] - 10
        Y_c = XY_c['y']
        O_c = (XY_c['o'] + 90)*2*np.pi/360
        D_c = (XY_c['dir'] + 90)*2*np.pi/360
        U_c = np.cos(O_c) 
        V_c = np.sin(O_c)
        M_c = np.cos(D_c)
        N_c = np.sin(D_c)
        
        X_r = XY_r['x'] - 10
        Y_r = XY_r['y']
        O_r = (XY_r['o'] + 90)*2*np.pi/360
        D_r = (XY_r['dir'] + 90)*2*np.pi/360
        U_r = np.cos(O_r) 
        V_r = np.sin(O_r)
        M_r = np.cos(D_r)
        N_r = np.sin(D_r)
        
        X_p = XY_p['x'] - 10
        Y_p = XY_p['y']
        X_pr = XY_pr['x'] - 10
        Y_pr = XY_pr['y']
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim(-10, 110)
        plt.ylim(0, 53.3)
        plt.scatter(X_c, Y_c, c = 'r')
        plt.scatter(X_r, Y_r, c = 'b')
        plt.scatter(X_p, Y_p, c = 'orange')
        plt.scatter(X_pr, Y_pr , c = 'c')
        plt.axvline(x = 0, c = 'w', linewidth = 1)
        plt.axvline(x = 100, c = 'w', linewidth = 1)
        plt.axvline(x = 10, c = 'w', linewidth = 1)
        plt.axvline(x = 20, c = 'w', linewidth = 1)
        plt.axvline(x = 30, c = 'w', linewidth = 1)
        plt.axvline(x = 40, c = 'w', linewidth = 1)
        plt.axvline(x = 50, c = 'w', linewidth = 1)
        plt.axvline(x = 60, c = 'w', linewidth = 1)
        plt.axvline(x = 70, c = 'w', linewidth = 1)
        plt.axvline(x = 80, c = 'w', linewidth = 1)
        plt.axvline(x = 90, c = 'w', linewidth = 1)
        plt.axvline(x = 5, c = 'w', linewidth = 0.5)
        plt.axvline(x = 15, c = 'w', linewidth = 0.5)
        plt.axvline(x = 25, c = 'w', linewidth = 0.5)
        plt.axvline(x = 35, c = 'w', linewidth = 0.5)
        plt.axvline(x = 45, c = 'w', linewidth = 0.5)
        plt.axvline(x = 55, c = 'w', linewidth = 0.5)
        plt.axvline(x = 65, c = 'w', linewidth = 0.5)
        plt.axvline(x = 75, c = 'w', linewidth = 0.5)
        plt.axvline(x = 85, c = 'w', linewidth = 0.5)
        plt.axvline(x = 95, c = 'w', linewidth = 0.5)
        ax.set_facecolor('#3f9b0b')
        
        speed_c, speed_r = self.compute_speed(no)
        
        ax.quiver( X_c, Y_c, U_c, V_c  , color = 'y', headwidth = 1, headlength  = 3, headaxislength = 3 )
        ax.quiver( X_r, Y_r, U_r, V_r, color = 'b', headwidth = 1, headlength  = 3, headaxislength = 3   )
        ax.quiver( X_r, Y_r, M_r, N_r, color = 'y', headwidth = 1, headlength  = 3, headaxislength = 3  )
        ax.quiver( X_c, Y_c, M_c, N_c, color = 'b', headwidth = 1, headlength  = 3, headaxislength = 3   )
        
    
    
    def export_frame(self, filename, no = 0):
        frame = self.sequences[no]
        XY_c = frame[frame['GSISID'].isin(self.coverage_player_id['GSISID'].tolist())]
        XY_r = frame[frame['GSISID'].isin(self.return_player_id['GSISID'].tolist())]
        XY_p = self.starting[self.starting['GSISID'] == self.P_ID]
        XY_pr = self.starting[self.starting['GSISID'] == self.PR_ID]
        X_c = XY_c['x'] - 10
        Y_c = XY_c['y']
        X_r = XY_r['x'] - 10
        Y_r = XY_r['y']
        X_p = XY_p['x'] - 10
        Y_p = XY_p['y']
        X_pr = XY_pr['x'] - 10
        Y_pr = XY_pr['y']
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.xlim(-10, 110)
        plt.ylim(0, 53.3)
        plt.scatter(X_c, Y_c, c = 'r')
        plt.scatter(X_r, Y_r, c = 'b')
        plt.scatter(X_p, Y_p, c = 'orange')
        plt.scatter(X_pr, Y_pr , c = 'c')
        plt.axvline(x = 0, c = 'w', linewidth = 1)
        plt.axvline(x = 100, c = 'w', linewidth = 1)
        plt.axvline(x = 10, c = 'w', linewidth = 1)
        plt.axvline(x = 20, c = 'w', linewidth = 1)
        plt.axvline(x = 30, c = 'w', linewidth = 1)
        plt.axvline(x = 40, c = 'w', linewidth = 1)
        plt.axvline(x = 50, c = 'w', linewidth = 1)
        plt.axvline(x = 60, c = 'w', linewidth = 1)
        plt.axvline(x = 70, c = 'w', linewidth = 1)
        plt.axvline(x = 80, c = 'w', linewidth = 1)
        plt.axvline(x = 90, c = 'w', linewidth = 1)
        plt.axvline(x = 5, c = 'w', linewidth = 0.5)
        plt.axvline(x = 15, c = 'w', linewidth = 0.5)
        plt.axvline(x = 25, c = 'w', linewidth = 0.5)
        plt.axvline(x = 35, c = 'w', linewidth = 0.5)
        plt.axvline(x = 45, c = 'w', linewidth = 0.5)
        plt.axvline(x = 55, c = 'w', linewidth = 0.5)
        plt.axvline(x = 65, c = 'w', linewidth = 0.5)
        plt.axvline(x = 75, c = 'w', linewidth = 0.5)
        plt.axvline(x = 85, c = 'w', linewidth = 0.5)
        plt.axvline(x = 95, c = 'w', linewidth = 0.5)
        ax.set_facecolor('#3f9b0b')
        if os.path.isdir('tmp'):
            os.chdir('tmp')
            fig.savefig( filename + '.png')
            os.chdir('..')
        else:
            os.mkdir('tmp')
            os.chdir('tmp')
            fig.savefig( filename + '.png')
            os.chdir('..')
        plt.close()
    
    def export_video(self, video_name, path = '', width = 120, height = 53.3):
        for i in range(len(self.sequences)):
            self.export_frame( 'img_'+ str(i).zfill(3), i)
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video=cv2.VideoWriter( path + video_name + '.avi', fourcc, 1,(width,height))    
        for i in range(1, len(self.sequences) + 1):
            img = cv2.imread('/tmp/' + 'img_' + str(i).zfill(3) + '.png')
            video.write(img)
        cv2.destroyAllWindows()
        video.release()
        
        os.chdir('tmp')
        for img in glob.glob('*.png'):
            os.remove(img)
        os.chdir('..')
        return None
    
    
        
        
### building a game object ###

class game(object):
    
    '''
    def __init__(self):
        self.GameKey = None
        self.Season = None
        self.Type = None
        self.Stadium = None
        self.StadiumType = None
        self.Weather = None
        self.Temperature = None
        self.Turf = None
    '''
        
        
    def __init__(self, gamekey, raw_data):
        play_data = raw_data[raw_data['GameKey'] == gamekey]
        self.GameKey = play_data['GameKey'][0]
        self.Season = play_data['Season_Year'][0]
        self.Type = play_data['Season_Type'][0]
        self.Stadium = play_data['Stadium'][0]
        self.StadiumType = play_data['StadiumType'][0]
        self.Weather = play_data['StadiumType'][0]
        self.Temperature = play_data['Temperature'][0]
        self.Turf = play_data['Turf'][0]
        self.playlist = []
        
        
        
    def feed(self, gamekey, raw_data):
        return None
    
    
test = play(6, 3236, NGS_2016_pre )
test.plot_starting()