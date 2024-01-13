from Data import Utils
from Data.GUIs import AugmentationsGUI

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

def empty_folder_content(folder_path, extensions=['.png', '.txt']):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(tuple(extensions)):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    return None

def main(augmentation_display=False, app_mode=False):
    """
    Main executable of the Data Augmentation Library app.

    Args:
        augmentation_display: Debugging tool. Default is [False]. 
            [True/False]: Plots a subplot object showing the augmentation results as images.
        app_mode: Runs the program in library or applicaiton mode. Default is [False]
            [True]: The program is run as an application, and the user can command any parameter.
            [False]: The program is run as a library, and all the parameters are from the DEFAULT config section.

    Inputs:
        augmentation_mode: Select the list of data augmentation to apply. Default is [0]
            [0]: Default
            [1]: All
            [2]: Custom
            [3]: Config
        rung_config: Select the paramters used in the transformation. Default is [DEFAULT]
            [DEFAULT]: Default config file parameters used, no randomness
            [Random]: Default parameters, with all continuous parameters set to random (eg: random rotation angle)
            [1]: Config n°1 parameters used
    """

    main_path = os.path.abspath('') + "/"
    empty_folder_content(folder_path=main_path+"augmented")
    run_config = "DEFAULT"

    # AUGMENTATION SELECTION
    if app_mode == True:
        augmentation_mode = int(input("> Select augmentation mode (default [0], all [1], custom [2], config [3]): ") or 0)
        if augmentation_mode not in [0, 1, 2, 3]:
            augmentation_mode = 0
        
        if augmentation_mode == 0:
            print(f"\033[1mDefault augmentation mode.\033[0m")
        elif augmentation_mode == 1:
            print(f"\033[1mAll possible augmentation mode.\033[0m")
        elif augmentation_mode == 2:
            print(f"\033[1mCustom augmentation mode.\033[0m")
        elif augmentation_mode == 3:
            print(f"\033[1mConfig augmentation mode.\033[0m")
            run_config = "GUI"
            gui = AugmentationsGUI.ConfigApp(config_path=main_path+"config.ini")
            gui.mainloop()

        else:
            augmentation_mode = 0
            print(f"\033[1mInvalid augmentation mode. Switched to default.\033[0m")
    else:
        augmentation_mode = 0

    # AUGMENTATION COMPUTATION
    dataAugmentationPipeline = Utils.pipelineController(main_path=main_path, mode=augmentation_mode, run_config=run_config)
    dataAugmentationPipeline.runPipeline()

    # AUGMENTATION DISPLAYING
    if augmentation_display:

        image_names= []
        image_names = [file[:-4] for file in os.listdir(main_path+"augmented") if file.endswith(".png")][0:20]

        num_images = len(image_names)   
        num_rows = int(np.sqrt(num_images))
        num_cols = num_images // num_rows
        image_names = [file[:-4] for file in os.listdir(main_path+"augmented") if file.endswith(".png")][0:num_rows*num_cols]

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6), squeeze=False)
        fig.subplots_adjust(wspace=0, hspace=0) 
        for subplot_idx, image_name in enumerate(image_names[:int(num_cols*num_rows)]):
            col, row = divmod(subplot_idx, num_rows)
            ax = axes[row, col]
            display_instance = Utils.Display(image_path=f"{main_path}/augmented/{image_name}.png",
                                                bbox_path=f"{main_path}/augmented/{image_name}.txt")
            display_instance.display(show=False, ax=ax)
        plt.tight_layout()
        plt.show()

    return 0

if __name__ == "__main__":
    main(augmentation_display=True, app_mode=True)


