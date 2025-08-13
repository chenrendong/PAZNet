import math
import numpy as np
import scipy.spatial as spatial
import scipy.ndimage.morphology as morphology

class Metirc():
    def __init__(self, real_mask, pred_mask, voxel_spacing):
        self.real_mask = real_mask
        self.pred_mask = pred_mask
        self.voxel_sapcing = voxel_spacing

        self.real_mask_surface_pts = self.get_surface(real_mask, voxel_spacing)
        self.pred_mask_surface_pts = self.get_surface(pred_mask, voxel_spacing)

        self.real2pred_nn = self.get_real2pred_nn()
        self.pred2real_nn = self.get_pred2real_nn()

    def get_surface(self, mask, voxel_spacing):

        kernel = morphology.generate_binary_structure(3, 2)
        surface = morphology.binary_erosion(mask, kernel) ^ mask

        surface_pts = surface.nonzero()

        surface_pts = np.array(list(zip(surface_pts[0], surface_pts[1], surface_pts[2])))
        return surface_pts * np.array(self.voxel_sapcing[::-1]).reshape(1, 3)

    def get_pred2real_nn(self):
        tree = spatial.cKDTree(self.real_mask_surface_pts)
        nn, _ = tree.query(self.pred_mask_surface_pts)

        return nn

    def get_real2pred_nn(self):
        tree = spatial.cKDTree(self.pred_mask_surface_pts)
        nn, _ = tree.query(self.real_mask_surface_pts)

        return nn

    def get_dice_coefficient(self):
        intersection = (self.real_mask * self.pred_mask).sum()
        union = self.real_mask.sum() + self.pred_mask.sum()

        return 2 * intersection / union, 2 * intersection, union

    def get_jaccard_index(self):
        intersection = (self.real_mask * self.pred_mask).sum()
        union = (self.real_mask | self.pred_mask).sum()

        return intersection / union

    def get_volumetric_error_rate(self):
        vol_real = float(self.real_mask.sum())
        vol_pred = float(self.pred_mask.sum())
        ver = abs(vol_real - vol_pred) / vol_real * 100
        return ver

    def get_hd95(self):
        all_distances = np.concatenate([self.real2pred_nn, self.pred2real_nn])
        hd95 = np.percentile(all_distances, 95)
        return hd95
