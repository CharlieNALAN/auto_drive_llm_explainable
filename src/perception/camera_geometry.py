#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def get_intrinsic_matrix(field_of_view_deg, image_width, image_height):
    """Calculate camera intrinsic matrix."""
    field_of_view_rad = field_of_view_deg * np.pi/180
    alpha = (image_width / 2.0) / np.tan(field_of_view_rad / 2.)
    Cu = image_width / 2.0
    Cv = image_height / 2.0
    return np.array([[alpha, 0, Cu],
                     [0, alpha, Cv],
                     [0, 0, 1.0]])

class CameraGeometry(object):
    """Camera geometry for lane detection."""
    
    def __init__(self, height=1.3, pitch_deg=5, image_width=1024, image_height=512, field_of_view_deg=45):
        self.height = height
        self.pitch_deg = pitch_deg
        self.image_width = image_width
        self.image_height = image_height
        self.field_of_view_deg = field_of_view_deg
        self.intrinsic_matrix = get_intrinsic_matrix(field_of_view_deg, image_width, image_height)
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)
        pitch = pitch_deg * np.pi/180
        cpitch, spitch = np.cos(pitch), np.sin(pitch)
        self.rotation_cam_to_road = np.array([[1,0,0],[0,cpitch,spitch],[0,-spitch,cpitch]])
        self.translation_cam_to_road = np.array([0,-self.height,0])
        self.trafo_cam_to_road = np.eye(4)
        self.trafo_cam_to_road[0:3,0:3] = self.rotation_cam_to_road
        self.trafo_cam_to_road[0:3,3] = self.translation_cam_to_road
        self.road_normal_camframe = self.rotation_cam_to_road.T @ np.array([0,1,0])

    def uv_to_roadXYZ_camframe(self,u,v):
        """Convert UV coordinates to road coordinates in camera frame."""
        uv_hom = np.array([u,v,1])
        Kinv_uv_hom = self.inverse_intrinsic_matrix @ uv_hom
        denominator = self.road_normal_camframe.dot(Kinv_uv_hom)
        return self.height*Kinv_uv_hom/denominator
    
    def camframe_to_roadframe(self,vec_in_cam_frame):
        """Convert camera frame coordinates to road frame coordinates."""
        return self.rotation_cam_to_road @ vec_in_cam_frame + self.translation_cam_to_road

    def uv_to_roadXYZ_roadframe(self,u,v):
        """Convert UV coordinates to road coordinates in road frame."""
        r_camframe = self.uv_to_roadXYZ_camframe(u,v)
        return self.camframe_to_roadframe(r_camframe)

    def uv_to_roadXYZ_roadframe_iso8855(self,u,v):
        """Convert UV coordinates to road coordinates in ISO 8855 frame."""
        X,Y,Z = self.uv_to_roadXYZ_roadframe(u,v)
        return np.array([Z,-X,-Y])

    def precompute_grid(self,dist=60):
        """Precompute grid for lane detection."""
        cut_v = int(self.compute_minimum_v(dist=dist)+1)
        xy = []
        for v in range(cut_v, self.image_height):
            for u in range(self.image_width):
                X,Y,Z= self.uv_to_roadXYZ_roadframe_iso8855(u,v)
                xy.append(np.array([X,Y]))
        xy = np.array(xy)
        return cut_v, xy

    def compute_minimum_v(self, dist):
        """Compute minimum v coordinate for a given distance."""
        trafo_road_to_cam = np.linalg.inv(self.trafo_cam_to_road)
        point_far_away_on_road = trafo_road_to_cam @ np.array([0,0,dist,1])
        uv_vec = self.intrinsic_matrix @ point_far_away_on_road[:3]
        uv_vec /= uv_vec[2]
        cut_v = uv_vec[1]
        return cut_v 