import os
import json
from typing import Any, Callable, Dict, List, TypeVar, Union
from kitti_common import (
    read_lines,
    parse_calib,
    get_label_anno,
)
from inspect import signature
import dill
import numpy as np


with open('params.json') as f:
    kitti_params = json.load(f)

detection_root = kitti_params['kitti_detection_root']
tracking_root = kitti_params['kitti_tracking_root']
segmentation_root = kitti_params['kitti_segmentation_root']

if 'compare_results_detection' in kitti_params:
    compare_results_detection = kitti_params['compare_results_detection']
    compare_results_detection_methods = compare_results_detection['methods']
else:
    compare_results_detection = None

if 'compare_results_tracking' in kitti_params:
    compare_results_tracking = kitti_params['compare_results_tracking']
    compare_results_tracking_methods = compare_results_tracking['methods']
else:
    compare_results_tracking = None

detection_folders = kitti_params['folders']['detection']
tracking_folders = kitti_params['folders']['tracking']
selected_split = kitti_params['selected_split']

T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')


def arrays_dict_element(dict: Dict[T, List[Any]], index: int):
    result = {}

    for key, val in dict.items():
        result[key] = val[index]
    
    return result


def map_dict(
    dict: Dict[T1, T2],
    func: Union[Callable[[T2], T3], Callable[[T2, T1], T3]]
):
    result = {}

    func_params_count = len(signature(func).parameters)

    if func_params_count == 1:
        for key, val in dict.items():
            result[key] = func(val)  # type: ignore
    elif func_params_count == 2:
        for key, val in dict.items():
            result[key] = func(val, key)  # type: ignore
    else:
        raise ValueError('Wrong reduce function')

    return result



def get_kitti_detection_files():

    result = {}

    for categoty, folder in detection_folders.items():
        path = detection_root + '/' + selected_split + '/' + folder
        result[categoty] = os.listdir(path) if os.path.isdir(path) else []
        result[categoty].sort()
        result[categoty] = list(map(
            lambda a: path + '/' + a,
            result[categoty]
        ))

    return result


def get_kitti_tracking_files():

    result = {}

    for categoty, folder in tracking_folders.items():
        path = tracking_root + '/' + selected_split + '/' + folder

        if categoty in ['calib', 'label']:
            result[categoty] = os.listdir(path) if os.path.isdir(path) else []
            result[categoty].sort()
            result[categoty] = list(map(
                lambda a: path + '/' + a,
                result[categoty]
            ))
        else:
            groups = os.listdir(path) if os.path.isdir(path) else []
            groups.sort()
            group_data = []

            for group in groups:
                local_path = path + '/' + group
                group_data.append(os.listdir(local_path) if os.path.isdir(local_path) else [])
                group_data[-1].sort()
                group_data[-1] = list(map(
                    lambda a: local_path + '/' + a,
                    group_data[-1]
                ))
        
            result[categoty] = group_data

    return result


def get_kitti_segmentation_files():

    result = []

    path = segmentation_root
    result = os.listdir(path) if os.path.isdir(path) else []
    result.sort()
    result = list(map(
        lambda a: path + '/' + a,
        result
    ))

    return result


def create_detection_kitti_info(files: dict, index):
    
    calib = parse_calib(
        read_lines(files['calib'])
    )

    label = get_label_anno(
        read_lines(files['label'])
    )


    result = {
        'image_idx': index,
        'velodyne_path': files['velodyne'],
        'img_path': files['image'],
        # 'img_shape': None,
        'calib/P0': calib['P'][0],
        'calib/P1': calib['P'][1],
        'calib/P2': calib['P'][2],
        'calib/P3': calib['P'][3],
        'calib/R0_rect': calib['R0_rect'],
        'calib/Tr_velo_to_cam': calib['Tr_velo_to_cam'],
        'calib/Tr_imu_to_velo': calib['Tr_velo_to_cam'],
        'annos': label
    }

    return result
        

def create_tracking_kitti_info(files: dict, index):
    
    calib = parse_calib(
        read_lines(files['calib'])
    )

    label_lines = read_lines(files['label'])

    selected_frame_lines = []

    for line in label_lines:

        parts = line.split(' ')
        frame = int(parts[0])

        if frame == index and parts[1] != '-1':
            selected_frame_lines.append(' '.join(parts[2:]))


    label = get_label_anno(
        selected_frame_lines
    )


    result = {
        'image_idx': index,
        'velodyne_path': files['velodyne'][index],
        'img_path': files['image'][index],
        # 'img_shape': None,
        'calib/P0': calib['P'][0],
        'calib/P1': calib['P'][1],
        'calib/P2': calib['P'][2],
        'calib/P3': calib['P'][3],
        'calib/R0_rect': calib['R0_rect'],
        'calib/Tr_velo_to_cam': calib['Tr_velo_to_cam'],
        'calib/Tr_imu_to_velo': calib['Tr_velo_to_cam'],
        'annos': label
    }

    return result


def create_segmentation_kitti_info(file_name: str):
    
    with open(file_name, 'rb') as f:
        segmented_points = dill.load(f)
        original_points = dill.load(f)
        ground_trurh_points = dill.load(f)

    segmented_points = segmented_points[:len(original_points)]
    ground_trurh_points = ground_trurh_points[:len(original_points)]

    if len(segmented_points.shape) == 1:
        segmented_points = segmented_points.reshape(-1, 1)

    points = np.concatenate([original_points, segmented_points], axis=-1)
    points = np.concatenate([points, ground_trurh_points], axis=-1)

    result = {
        'points': points
    }

    return result


def get_compare_detection_annotation(method, index):
    file_name = (
        compare_results_detection['root'] + '/' + method + '/'
        + selected_split + '/' + '{:06d}'.format(index) + '.txt'
    )

    lines = []

    if os.path.exists(file_name):
        lines = read_lines(file_name)
    
    label = get_label_anno(
        lines
    )

    return label


def get_compare_tracking_annotation(method, group, index):

    annotation_format = compare_results_tracking['format']

    if annotation_format == "ab3dmot":

        type_by_number = {1:'Pedestrian', 2:'Car', 3:'Cyclist'}

        file_name = (
            compare_results_tracking['root'] + '/' + method + '/'
            + selected_split + '/' + '{:04d}'.format(group) + '.txt'
        )

        lines = []

        if os.path.exists(file_name):
            lines = read_lines(file_name)
            selected_lines = []

            for i in range(len(lines)):
                line = lines[i]
                elements = line.replace('\n','').split(',')

                if len(elements) <= 1:
                    continue

                (
                    id, t, a1, a2, a3, a4,
                    b, c1, c2, c3, c4, c5, c6,
                    alpha, beta
                ) = elements



                new_elements = [
                    type_by_number[int(t)],
                    0,
                    0,
                    beta,
                    a1,
                    a2,
                    a3,
                    a4,
                    c1,
                    c2,
                    c3,
                    c4,
                    c5,
                    c6,
                    alpha,
                    b
                ]

                new_line = " ".join([str(a) for a in new_elements])

                current_index = int(id)

                if index == current_index:
                    selected_lines.append(new_line)

            
            lines = selected_lines

        label = get_label_anno(
            lines
        )

        return label
    else:
        raise ValueError("Unsupported tracking format: " + annotation_format)