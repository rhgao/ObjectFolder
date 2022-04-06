'''
GelSight tactile render with taxim

Zilin Si (zsi@andrew.cmu.edu)
Last revision: March 2022
'''

import os
from os import path as osp
import numpy as np
import cv2

from basics import sensorParams as psp
from basics.CalibData import CalibData

class TaximRender:

    def __init__(self, calib_path):
        # taxim calibration files
        # polytable
        calib_data = osp.join(calib_path, "polycalib.npz")
        self.calib_data = CalibData(calib_data)
        # raw calibration data
        rawData = osp.join(calib_path, "dataPack.npz")
        data_file = np.load(rawData, allow_pickle=True)
        self.f0 = data_file['f0']
        ## tactile image config
        bins = psp.numBins
        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        self.A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h*psp.w)]).T
        binm = bins - 1
        self.x_binr = 0.5*np.pi/binm # x [0,pi/2]
        self.y_binr = 2*np.pi/binm # y [-pi, pi]

        # load depth bg
        self.bg_depth = np.load(osp.join(calib_path,"depth_bg.npy"), allow_pickle=True)
        # load tactile bg
        self.real_bg = np.load(osp.join(calib_path,"real_bg.npy"), allow_pickle=True)

    def correct_height_map(self, height_map):
        # move the center of depth to the origin
        height_map = (height_map-psp.cam2gel) * -1000 / psp.pixmm
        return height_map

    def padding(self, img):
        # pad one row & one col on each side
        if len(img.shape) == 2:
            return np.pad(img, ((1, 1), (1, 1)), 'edge')
        elif len(img.shape) == 3:
            return np.pad(img, ((1, 1), (1, 1), (0, 0)), 'edge')

    def generate_normals(self, height_map):
        # from height map to gradient magnitude & directions

        [h,w] = height_map.shape
        center = height_map[1:h-1,1:w-1] # z(x,y)
        top = height_map[0:h-2,1:w-1] # z(x-1,y)
        bot = height_map[2:h,1:w-1] # z(x+1,y)
        left = height_map[1:h-1,0:w-2] # z(x,y-1)
        right = height_map[1:h-1,2:w] # z(x,y+1)
        dzdx = (bot-top)/2.0
        dzdy = (right-left)/2.0

        mag_tan = np.sqrt(dzdx**2 + dzdy**2)
        grad_mag = np.arctan(mag_tan)
        invalid_mask = mag_tan == 0
        valid_mask = ~invalid_mask
        grad_dir = np.zeros((h-2,w-2))
        grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask]/mag_tan[valid_mask], dzdy[valid_mask]/mag_tan[valid_mask])

        grad_mag = self.padding(grad_mag)
        grad_dir = self.padding(grad_dir)
        return grad_mag, grad_dir

    def render(self, depth, press_depth):

        depth = self.correct_height_map(depth)
        height_map = depth.copy()

        ## generate contact mask
        pressing_height_pix = press_depth * 1000 / psp.pixmm
        contact_mask = (height_map-(self.bg_depth)) > pressing_height_pix * 0.2

        # smooth out the soft contact
        zq_back = height_map.copy()
        kernel_size = [11,5]
        for k in range(len(kernel_size)):
            height_map = cv2.GaussianBlur(height_map.astype(np.float32),(kernel_size[k],kernel_size[k]),0)
            height_map[contact_mask] = zq_back[contact_mask]
        # height_map = cv2.GaussianBlur(height_map.astype(np.float32),(5,5),0)

        # generate gradients
        grad_mag, grad_dir = self.generate_normals(height_map)

        # simulate raw image
        sim_img_r = np.zeros((psp.h,psp.w,3))
        idx_x = np.floor(grad_mag/self.x_binr).astype('int')
        idx_y = np.floor((grad_dir+np.pi)/self.y_binr).astype('int')

        params_r = self.calib_data.grad_r[idx_x,idx_y,:]
        params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x,idx_y,:]
        params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x,idx_y,:]
        params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

        est_r = np.sum(self.A * params_r,axis = 1)
        est_g = np.sum(self.A * params_g,axis = 1)
        est_b = np.sum(self.A * params_b,axis = 1)
        sim_img_r[:,:,0] = est_r.reshape((psp.h,psp.w))
        sim_img_r[:,:,1] = est_g.reshape((psp.h,psp.w))
        sim_img_r[:,:,2] = est_b.reshape((psp.h,psp.w))

        # add back ground
        tactile_img = sim_img_r + self.real_bg
        tactile_img = np.clip(tactile_img, 0, 255)

        return height_map, contact_mask, tactile_img

#if __name__ == "__main__":

    # define the press depth, and get the depth map from touch net.
    # taxim = TaximRender(calib_path)
    # height_map, contact_mask, tactile_img = taxim.render(depth, press_depth)
