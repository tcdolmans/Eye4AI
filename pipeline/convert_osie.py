"""
 * @author [Tenzing Dolmans]
 * @email [t.c.dolmans@gmail.com]
 * @create date 2023-05-11 12:06:29
 * @modify date 2023-05-11 12:06:29
 * @desc [description]
"""
import os
# import h5py
import glob
import torch
import pandas as pd
from convert_sets import replace_nans, downsample


def convert_et_data(et_folder, max_length=300):
    """
    Load the eye-tracking data from the specified folder.
    1. List files that are present in et_folder
    2. Per file Load the eye-tracking data from the .h5 file
    3. Extract gaze and msg data: gaze = pd.read_hdf(file, 'gaze'), msg = pd.read_hdf(file, 'msg')
    4. Make a list that contains img names "1001.jpg" to "1700.jpg"
    5. Loop over the list made in step 4 for 'msg' and grab the "system_time_stamp" as start time
    when "onset_img_name" and end time when "offset_img_name" matches the img name
    6. In 'gaze', find the corresponding rows for the start and end times using the
    start and end times from step 5
    7. From these rows, collect the following:
        - "system_time_stamp"
        - "left_gaze_point_on_display_area_x"
        - "left_gaze_point_on_display_area_y"
        - "right_gaze_point_on_display_area_x"
        - "right_gaze_point_on_display_area_y"
        - "left_pupil_diameter"
        - "right_pupil_diameter"
    8. Average the left and right gaze points for x and y, and pupil diameters into
    x and y coordinate and pupil diameter
    9. Create a tensor with the following columns:
        - "system_time_stamp"
        - "gaze_point_x"
        - "gaze_point_y"
        - "pupil_diameter"
    10. Save the tensor as a file in the et_folder named "participant_number_stimulus_number.pt"
    """
    # 1. List .h5 files that are present in et_folder
    et_files = sorted(glob.glob(os.path.join(et_folder, '*.h5')))
    # 4. Make a list that contains img names "1001.jpg" to "1700.jpg"
    img_names = [f"{i}.jpg" for i in range(1001, 1701)]

    for et_file in et_files:
        # 2. Per file Load the eye-tracking data from the .h5 file
        gaze = pd.read_hdf(et_file, 'gaze')
        msg = pd.read_hdf(et_file, 'msg')
        for img_name in img_names:
            # 5. Loop over the list made in step 4 for 'msg' and grab the "system_time_stamp" as
            # start time when "onset_img_name" and end time when "offset_img_name" matches
            start_time_idx = msg.loc[msg['msg'].str.contains(
                f'onset.*{img_name}')].first_valid_index()
            end_time_idx = msg.loc[msg['msg'].str.contains(
                f'offset.*{img_name}')].first_valid_index()

            # Check whether the start and end times are empty, not every image is used as a stimulus
            if start_time_idx is None or end_time_idx is None:
                continue

            start_time = msg.loc[start_time_idx, 'system_time_stamp']
            end_time = msg.loc[end_time_idx, 'system_time_stamp']

            # 6. In 'gaze', find the corresponding rows for the start and end times
            gaze_rows = gaze.loc[(gaze['system_time_stamp'] >= start_time)
                                 & (gaze['system_time_stamp'] <= end_time)]

            # 7. From these rows, collect the required columns
            gaze_data = gaze_rows[[
                'system_time_stamp',
                'left_gaze_point_on_display_area_x',
                'left_gaze_point_on_display_area_y',
                'right_gaze_point_on_display_area_x',
                'right_gaze_point_on_display_area_y',
                'left_pupil_diameter',
                'right_pupil_diameter'
            ]]

            # 8. Average the left and right gaze points for x and y, and pupil diameters into x and
            # y coordinate and pupil diameter
            # TODO: Check whether the gaze points are valid
            # TODO: DS and CheckNan from utils
            gaze_out = pd.DataFrame(columns=['system_time_stamp', 'gaze_point_x',
                                             'gaze_point_y', 'pupil_diameter'])
            gaze_out['system_time_stamp'] = gaze_data['system_time_stamp']
            gaze_out['gaze_point_x'] = (gaze_data['left_gaze_point_on_display_area_x'] +
                                        gaze_data['right_gaze_point_on_display_area_x']) / 2
            gaze_out['gaze_point_y'] = (gaze_data['left_gaze_point_on_display_area_y'] +
                                        gaze_data['right_gaze_point_on_display_area_y']) / 2
            gaze_out['pupil_diameter'] = (gaze_data['left_pupil_diameter'] +
                                          gaze_data['right_pupil_diameter']) / 2

            # 9. Create a tensor with the following columns:
            tensor_data = gaze_out[['system_time_stamp', 'gaze_point_x',
                                    'gaze_point_y', 'pupil_diameter']].to_numpy()
            et_tensor = torch.tensor(tensor_data, dtype=torch.float32)
            et_tensor = replace_nans(et_tensor, threshold=0.69)
            if et_tensor.size(0) > max_length:
                et_tensor = et_tensor[:max_length]
            elif et_tensor.size(0) < max_length:
                pad_size = max_length - et_tensor.size(0)
                padding = torch.zeros(pad_size, et_tensor.size(1), dtype=torch.float32)
                et_tensor = torch.cat((et_tensor, padding), dim=0)
            # 10. Save the tensor as a file in the et_folder named appropriately
            p_num = os.path.splitext(os.path.basename(et_file))[0][-4:]
            stim_num = os.path.splitext(img_name)[0]
            torch.save(et_tensor, os.path.join(et_folder, 'osie_tensors', f"{p_num}_{stim_num}.pt"))

        print("Done with file: ", p_num)


if __name__ == "__main__":
    base_path = os.getcwd()
    et_folder = os.path.join(base_path, 'pipeline/osieData')
    convert_et_data(et_folder)
