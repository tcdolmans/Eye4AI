import os
# import h5py
import glob
import torch
import pandas as pd


def convert_et_data(et_folder):
    """
    Load the eye-tracking data from the specified folder.
    1. List files that are present in et_folder
    2. Per file Load the eye-tracking data from the .h5 file
    3. Extract gaze and msg data: gaze = pd.read_hdf(file, 'gaze'), msg = pd.read_hdf(file, 'msg')
    4. Make a list that contains img names "1001.jpg" to "1700.jpg"
    5. Loop over the list made in step 4 for 'msg' and grab the "system_time_stamp" as start time when "onset_img_name" and end time when "offset_img_name" matches the img name
    6. In 'gaze', find the corresponding rows for the start and end times using the start and end times from step 5
    7. From these rows, collect the following:
        - "system_time_stamp"
        - "left_gaze_point_on_display_area_x"
        - "left_gaze_point_on_display_area_y"
        - "right_gaze_point_on_display_area_x"
        - "right_gaze_point_on_display_area_y"
        - "left_pupil_diameter"
        - "right_pupil_diameter"
    8. Average the left and right gaze points for x and y, and pupil diameters into x and y coordinate and pupil diameter
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
        print("HERE!!!!!!!", et_file)
        # 2. Per file Load the eye-tracking data from the .h5 file
        # with h5py.File(et_file, 'r') as file:
        #     # 3. Extract gaze and msg data
        #     gaze = pd.read_hdf(file, 'gaze')
        #     msg = pd.read_hdf(file, 'msg')

        gaze = pd.read_hdf(et_file, 'gaze')
        msg = pd.read_hdf(et_file, 'msg')
        for img_name in img_names:
            # 5. Loop over the list made in step 4 for 'msg' and grab the "system_time_stamp" as
            # start time when "onset_img_name" and end time when "offset_img_name" matches
            start_time = msg.loc[msg['msg'].str.contains(f'onset.*{img_name}')]['system_time_stamp'].values
            end_time = msg.loc[msg['msg'].str.contains(f'offset.*{img_name}')]['system_time_stamp'].values
            print(start_time, end_time)
            # 6. In 'gaze', find the corresponding rows for the start and end times using the start
            # and end times from step 5
            print(len(gaze.loc[(gaze['system_time_stamp'])]), len(start_time))
            gaze_rows = gaze.loc[(gaze['system_time_stamp'] >= start_time) & (gaze['system_time_stamp'] <= end_time)]

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
            gaze_data['gaze_point_x'] = (gaze_data['left_gaze_point_on_display_area_x'] +
                                         gaze_data['right_gaze_point_on_display_area_x']) / 2
            gaze_data['gaze_point_y'] = (gaze_data['left_gaze_point_on_display_area_y'] +
                                         gaze_data['right_gaze_point_on_display_area_y']) / 2
            gaze_data['pupil_diameter'] = (gaze_data['left_pupil_diameter'] +
                                           gaze_data['right_pupil_diameter']) / 2

            # 9. Create a tensor with the following columns:
            tensor_data = gaze_data[['system_time_stamp', 'gaze_point_x',
                                     'gaze_point_y', 'pupil_diameter']].to_numpy()
            et_tensor = torch.tensor(tensor_data, dtype=torch.float32)

            # 10. Save the tensor as a file in the et_folder named appropriately
            p_num = os.path.splitext(os.path.basename(et_file))[0]  # NOTE: Not correct
            stim_num = os.path.splitext(img_name)[0]  # NOTE: Not correct
            torch.save(et_tensor, os.path.join(et_folder, f"{p_num}_{stim_num}.pt"))


if __name__ == "__main__":
    et_folder = 'C:\\Users\\Tenzing Dolmans\\OneDrive - O365 Turun yliopisto\\Documents\\1. Human Neuroscience\\Thesis\\Code\\Eye4AI\pipeline\\osieData'
    convert_et_data(et_folder)
