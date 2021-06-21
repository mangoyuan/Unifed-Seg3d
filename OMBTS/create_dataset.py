# -*- coding: utf-8 -*-
from collections import OrderedDict
import SimpleITK as sitk
import os
import yaml
from shutil import copyfile
import json
import argparse


def create_single_modal_dataset(task_dir, output_dir, dyaml):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # create the sub output dirs
    img_dir = os.path.join(output_dir, 'imagesTr')
    lab_dir = os.path.join(output_dir, 'labelsTr')
    img_dir_te = os.path.join(output_dir, 'imagesTs')
    lab_dir_te = os.path.join(output_dir, 'labelsTs')
    for path in [task_dir, img_dir, lab_dir, img_dir_te, lab_dir_te]:
        if not os.path.exists(path):
            os.mkdir(path)

    # Read the configure files.
    with open(dyaml, 'r') as f:
        caseid_model = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(task_dir, 'dataset.json'), 'r') as f:
        djson = json.load(f)
    modal2index = {m: int(i) for i, m in djson['modality'].items()}

    train_list = []
    test_list = []
    save_dir = {'train': [img_dir, lab_dir], 'test': [img_dir_te, lab_dir_te]}
    for phase in save_dir.keys():
        for caseid in sorted(caseid_model[phase]):
            fname = '{}.nii.gz'.format(caseid)
            if phase == 'test':
                test_list.append(fname)
            else:
                train_list.append(fname)

            # Select one modality for volumes.
            selected_m = caseid_model[phase][caseid]
            selected_index = modal2index[selected_m]

            image_p = os.path.join(task_dir, 'imagesTr', fname)
            itk_img = sitk.ReadImage(image_p)

            img = sitk.GetArrayFromImage(itk_img)
            img = img[selected_index, :, :, :]

            out = sitk.GetImageFromArray(img)
            sitk.WriteImage(out, os.path.join(save_dir[phase][0], fname))

            # Save Labels.
            src_p = os.path.join(task_dir, 'labelsTr', fname)
            dst_p = os.path.join(save_dir[phase][1], fname)
            copyfile(src_p, dst_p)
            print('Read {} in {} phase, save in {} and {}.'.format(fname, phase, save_dir[phase][0], save_dir[phase][1]))

    json_dict = OrderedDict()
    json_dict['name'] = "OMBTS"
    json_dict['description'] = "One Modality Brain Tumor Segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "mri"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "edema",
        "2": "non-enhancing tumor",
        "3": "enhancing tumor",
    }

    json_dict['numTraining'] = len(train_list)
    json_dict['numTest'] = len(test_list)
    json_dict['training'] = [{'image': "./imagesTr/{}".format(i), "label": "./labelsTr/{}".format(i)} for i in train_list]
    json_dict['test'] = ["./imagesTs/%s" % i for i in test_list]
    print(json_dict)
    with open(os.path.join(output_dir, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--task_dir', type=str)
    parse.add_argument('--out_dir', type=str)
    parse.add_argument('--c2m', type=str)

    args = parse.parse_args()
    create_single_modal_dataset(args.task_dir, args.out_dir, args.c2m)

