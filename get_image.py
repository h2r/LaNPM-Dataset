# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

#This BD code is heavily edited by Ahmed Jaafar

import cv2
import numpy as np
import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)


def transform(image, dtype, num_bytes):
    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    return img


def capture(robot, format):
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)
    img_sources = ['frontleft_fisheye_image','frontright_fisheye_image']
    depth_sources = ['frontright_depth', 'frontleft_depth']
    imgs = []

    if format == "PIXEL_FORMAT_RGB_U8":

        pixel_format = pixel_format_string_to_enum(format)
        
        image_request = [
            build_image_request(source, pixel_format=pixel_format)
            for source in img_sources
        ]
        image_responses = image_client.get_image(image_request)

        for image in image_responses:
            num_bytes = 3
            dtype = np.uint8
            # extension = '.jpg'
            img = transform(image, dtype, num_bytes)
            img = np.rot90(img, k=3)
            imgs.append[img]

    else:
        pixel_format = pixel_format_string_to_enum(format)
        
        image_request = [
            build_image_request(source, pixel_format=pixel_format)
            for source in depth_sources
        ]
        image_responses = image_client.get_image(image_request)

        for image in image_responses:
            num_bytes = 1 
            dtype = np.uint16
            # extension = '.png'
            img = transform(image, dtype, num_bytes)
            img = np.rot90(img, k=3)
            imgs.append[img]

            imgs.append[img]
    
    return imgs