# PyYel libraries
from .Augmentations import ImageAugmentation
from ..constants import * 
from .guis.AugmentationsGUI import ConfigApp

# Utils
import numpy as np
import pandas as pd
import cv2
import configparser
import ast
import os
from tqdm import tqdm

# Files parsing
import xml.etree.ElementTree as ET
import json
# import tensorflow as tf

__all__ = ["Pipeliner", "ImageAugmentationPipeline"]

class Pipeliner():
    """
    Wrapper of the utility pipelines.
    Allows an easier utilization of the library, with simplified inputs.

    Args:
        data_type: the type of data to process.
            [image]: the data expected are .png or .jpg images located in the folder /input_path/. Supports .txt, 
            .xml and .json labels files (boundary boxes).
            [table]: the data expected are .txt or .csv tables located in the folder /input_path/.
        input_path: the location of the orignal data folder. Datapoints should be directly accessible (no sub-folder).
        output_path: the location of the augmented data folder. Augmented datapoints will be saved there.
        mode: the execution mode.
            [0]: Default configuration
            [1]: Full configuration
            [2]: Command prompt pilotable configuration
            [3]: GUI interface configuration
    """

    def __init__(self, data_type, input_path, output_path, mode) -> None:
        
        self.data_type = data_type
        self.input_path = input_path
        self.output_path = output_path
        self.mode = mode

        if self.data_type == "image":
            ImageAugmentationPipeline(input_path=self.input_path,
                                      output_path=self.output_path,
                                      mode=self.mode).runPipeline()
        elif self.data_type == "table":
            raise ValueError("data_type=table: Not supported yet")
        else:
            None
class ImageAugmentationPipeline():
    """
    A utility to process the datapoints augmentations.

    Attributes:
        data_folder: The path to the input folder.
        augmented_folder: The path to the output folder.
        library_folder: The path to the Data Augmentation Library scripts and methods.

    Args:
        main_path: The location of the application on the user's system.
        mode: The augmentation listing to use. Default is Default [0]
            [0]: Default (rotation, gray_scale)
            [1]: All (applies all the available augmentations)
            [2]: Custom (allows the user to enter his list of transform)
            [3]: Config (reads augmentation list from the config file)
        data_type: Augmentation list depends on the datapoints to augment. Default is [image].
            [image]: Available augmentations include image augmentation methods.
        run_config: Config setup to use. Default is [DEFAULT]
            [DEFAULT]: Default config file parameters used, no randomness
            [Random]: Default parameters, with all continuous parameters set to random (eg: random rotation angle)
            [1]: Config n°1 parameters used
        debug: Runs debuging console outputs. Default is [False].
    """

    def __init__(self, input_path, output_path, mode=0) -> None:

        self.mode = mode

        self.data_folder = input_path
        self.augmented_folder = output_path

        self.library_folder = DIR_DATA_PATH
        self.file_folder = os.path.join(DIR_DATA_PATH, __name__)

    def runPipeline(self):
        main_dataAugmentationPipeline = [
            self.selectMode(),
            self._load_config(),
            self._read_config(),
            self.filesListing(),
            self.filesDataframing(),
            self.augmentationsListing(),
            self.augmentationsPipelining(),
            self.augmentationsApplying(),
            self._reset_config(),
        ]

        for main_pipeline_step in main_dataAugmentationPipeline:
            main_pipeline_step

        return None
    
    def selectMode(self):
        if self.mode == 0:
            print(f"\033[1mDefault augmentation mode.\033[0m")
            self.run_config = "DEFAULT"
        elif self.mode == 1:
            print(f"\033[1mAll possible augmentation mode.\033[0m")
            self.run_config = "GUI"
        elif self.mode == 2:
            print(f"\033[1mCustom augmentation mode.\033[0m")
            self.run_config = "GUI"
        elif self.mode == 3:
            print(f"\033[1mConfig augmentation mode.\033[0m")
            self.run_config = "GUI"
            gui = ConfigApp(config_path=os.path.join(DIR_DATA_CONFIGS_PATH, "AugmentationsConfig.ini"))
            gui.mainloop()

    # PIPELINE STEPS
    def filesListing(self):
        """
        Pipeline step.\n
        Retreives a list of the to-be-augmented files in the /data/ folder, and its labels files.  
        Returns the to-augment-data list of datapoint files and list of datapoints labels.
        """

        file_list = os.listdir(self.data_folder)

        self.data_list = []
        self.data_list = [file for file in file_list if file.endswith(".jpg")]
        self.data_list = self.data_list + [file for file in file_list if file.endswith(".png")]

        self.labels_list = []
        self.labels_list = [file for file in file_list if file.endswith(".txt")]
        self.labels_list = self.labels_list + [file for file in file_list if file.endswith(".xml")]
        self.labels_list = self.labels_list + [file for file in file_list if file.endswith(".json")]
        self.labels_list = self.labels_list + [file for file in file_list if file.endswith(".tfrecord")]

        if len(self.data_list) == 0:
            raise ValueError("No image to augment found")
            
        return self.data_list, self.labels_list


    def filesDataframing(self):
        """
        Pipeline step.
        Generates a dataframe featuring standardized datapoints infos. Saves it as .csv file.
        Returns the pandas dataframe object.
        """
        
        if self.data_type == "image":
                
            columns=["Path", "Name", "Extension", "Labels", "Positions"]
            df_files = pd.DataFrame([], columns=columns)
            for file in self.data_list:
                file_path, file_name, file_extension = self._fileNaming(file=file)

                if f"{file_name}.txt" in self.labels_list:
                    labels, locations, = self._parse_txt(f"{file_path}/{file_name}")
                    df = pd.DataFrame([[file_path, file_name, file_extension, labels, locations]], columns=columns)
                    df_files = pd.concat([df_files, df], ignore_index=True)

                elif f"{file_name}.xml" in self.labels_list:
                    labels, locations = self._parse_xml(f"{file_path}/{file_name}")
                    df = pd.DataFrame([[file_path, file_name, file_extension, labels, locations]], columns=columns)
                    df_files = pd.concat([df_files, df], ignore_index=True)

                elif f"{file_name}.json" in self.labels_list:
                    labels, locations = self._parse_json(f"{file_path}/{file_name}")
                    df = pd.DataFrame([[file_path, file_name, file_extension, labels, locations]], columns=columns)
                    df_files = pd.concat([df_files, df], ignore_index=True)

                # elif f"{file_name}.tfrecords" in self.labels_list:
                #     labels, locations = self._parse_tfrecord(f"{file_path}/{file_name}")
                #     df = pd.DataFrame([[file_path, file_name, file_extension, labels, locations]], columns=columns)
                #     df_files = pd.concat([df_files, df], ignore_index=True)

                else: # if there is no boundary box file
                    labels = file_name # label may be the name of the file, otherwise it's easy to change in the dataframe
                    # locations = [[0.5, 0.5, 1, 1]] # we define a boundary box that covers the whole image (compatibility purpose)
                    locations = None
                    df = pd.DataFrame([[file_path, file_name, file_extension, labels, locations]], columns=columns)
                    df_files = pd.concat([df_files, df], ignore_index=True)

            df_files.index.name = "Input_Index"
            df.reset_index(drop=True, inplace=True)
            df_files.to_csv(os.path.join(self.augmented_folder, "datapoints.csv"))
            self.df_files = df_files

        else:
            None

        return self.df_files

    def augmentationsListing(self):
        """
        Pipeline step.
        Allows the user to enter a list of augmentations to perform. Confirm with [confirm] input.
        Default is ["rotation", "gray_scale"].
        """
        
        self.possible_augmentations = [
            "rotation", 
            "horizontal_flip",
            "vertical_flip",
            "gray_scale", 
            "channels_swap",
            "cut",
            "brightness",
            "noise",
            "contrast",
            "colour_inversion",
            "zoom",
            "blur",
            "edges",
        ]

        self.augmentation_list = []
        if self.mode == 0:
            # Default data augmentation pre selection
            self.augmentation_list = ["gray_scale", "rotation"]

        elif self.mode == 1:
            # All possible augmentations
            self.augmentation_list = self.possible_augmentations

        elif self.mode == 2:
            # User's custom data augmentation selection
            augmentation_input = ""
            while augmentation_input != "confirm":
                print("Select augmentations among:", self.possible_augmentations)
                print("Confirm selection with [confirm]")

                augmentation_input = input("> Enter <augmentation_name> or <confirm>: ")
                if augmentation_input in self.possible_augmentations:
                    self.augmentation_list.append(augmentation_input)

                elif augmentation_input == "confirm":
                    continue

                else:
                    print(f"\033[1mAugmentation type not supported.\033[0m")

        elif self.mode == 3:
            # Config file augmentation selection
            self.augmentation_list = ast.literal_eval(self.config.get(self.run_config, "augmentation_list"))

        # If no input, switch to default mode
        if self.augmentation_list == []:
            self.augmentation_list = ["gray_scale", "rotation"]
            self.run_config = "DEFAULT"
        print("Selected augmentation types are: ", f"\033[1m{self.augmentation_list}\033[0m")
        print("Selected config is: ", f"\033[1m{self.run_config}\033[0m")

        return self.augmentation_list

    def augmentationsPipelining(self):
        """
        Pipeline step.\n
        Generates two ImageAugmentation pipelines to handle standard data augmentation, and 
        augmentations that require to perform a new labeling task. \n
        Data is automatically sent into the right pipeline according to the infos retreived from the 
        augmented_datapoint_report.csv dataframe. \n
        Pipelines save the augmented data into the /augmented/ folder, alongside its augmented labels txt.
        """
        
        # Dictionnaries of available augmentation methods 
        simple_augmentation_dict = {
            "rotation": ImageAugmentation.imageRotation,
            "horizontal_flip": ImageAugmentation.imageHorizontalFlip, 
            "vertical_flip": ImageAugmentation.imageVerticalFlip, 
            "gray_scale": ImageAugmentation.imageGrayScale,
            "channels_swap": ImageAugmentation.imageChannelsSwap,
            "cut": ImageAugmentation.imageCut,
            "brightness": ImageAugmentation.imageBrightness,
            "contrast": ImageAugmentation.imageContrast,
            "noise": ImageAugmentation.imageNoise,
            "zoom": ImageAugmentation.imageZoom,
            "colour_inversion": ImageAugmentation.imageColourInversion,
            "blur": ImageAugmentation.imageBlur,
            "edges": ImageAugmentation.imageEdges,
        }

        bbox_augmentation_dict= {
            "rotation": [ImageAugmentation.imageRotation, ImageAugmentation.labelsRotation],
            "horizontal_flip": [ImageAugmentation.imageHorizontalFlip, ImageAugmentation.labelsHorizontalFlip],
            "vertical_flip": [ImageAugmentation.imageVerticalFlip, ImageAugmentation.labelsVerticalFlip],
            "gray_scale": ImageAugmentation.imageGrayScale,
            "channels_swap": ImageAugmentation.imageChannelsSwap,
            "cut": ImageAugmentation.imageCut,
            "brightness": ImageAugmentation.imageBrightness,
            "contrast": ImageAugmentation.imageContrast,
            "noise": ImageAugmentation.imageNoise,
            "zoom": [ImageAugmentation.imageZoom, ImageAugmentation.labelsZoom],
            "colour_inversion": ImageAugmentation.imageColourInversion,
            "blur": ImageAugmentation.imageBlur,
            "edges": ImageAugmentation.imageEdges,
        }

        # Pipeline generated according to the selected augmentation methods
        self.simple_augmentation_pipeline = []
        self.bbox_augmentation_pipeline = []
        for augmentation_type in self.augmentation_list:
            self.simple_augmentation_pipeline.append(simple_augmentation_dict[augmentation_type])
            self.bbox_augmentation_pipeline.append(bbox_augmentation_dict[augmentation_type])

        self.simple_df_files = self.df_files[self.df_files["Positions"].isnull()] # Datapoint without localisation
        self.bbox_df_files = self.df_files[~self.df_files["Positions"].isnull()] # Datapoint with at least 1 boundary box

        return None

    def augmentationsApplying(self):

        # Augments regular images
        for data_idx, data_row in tqdm(self.simple_df_files.iterrows(), total=len(self.bbox_df_files), desc="Processing simple images augments"):

            data_row_path = f"{data_row['Path']}/{data_row['Name']}.{data_row['Extension']}"
            data_labels = data_row["Labels"]
            data_locations = data_row["Positions"]
            data_array = np.array(cv2.imread(data_row_path))

            image_augmentation_instance = ImageAugmentation(data=data_array, 
                                                            labels=data_labels, 
                                                            positions=data_locations,
                                                            config=self.config,
                                                            run_config=self.run_config)
            for step in self.simple_augmentation_pipeline:
                step(image_augmentation_instance)
                
                augmented_data = image_augmentation_instance.augmented_data
                current_augment = image_augmentation_instance.current_augment

                cv2.imwrite(f"{self.augmented_folder}/{data_row['Name']}__{current_augment}.png", augmented_data)


        # Augments images with boundary boxes
        for data_idx, data_row in tqdm(self.bbox_df_files.iterrows(), total=len(self.bbox_df_files), desc="Processing bbox images augments"):

            data_row_path = f"{data_row['Path']}/{data_row['Name']}.{data_row['Extension']}"
            data_labels = data_row["Labels"]
            data_locations = data_row["Positions"]
            data_array = np.array(cv2.imread(data_row_path)) 

            image_augmentation_instance = ImageAugmentation(data=data_array, 
                                                            labels=data_labels, 
                                                            positions=data_locations,
                                                            config=self.config,
                                                            run_config=self.run_config)
            for step_idx, step in enumerate(self.bbox_augmentation_pipeline):
                if type(step) == list: # If label rotation is needed
                    for sub_step in step:
                        sub_step(image_augmentation_instance)
            
                    augmented_data = image_augmentation_instance.augmented_data
                    augmented_txt = image_augmentation_instance.augmented_txt
                    current_augment = image_augmentation_instance.current_augment

                    cv2.imwrite(f"{self.augmented_folder}/{data_row['Name']}__{current_augment}.png", augmented_data)
                    np.savetxt(f"{self.augmented_folder}/{data_row['Name']}__{current_augment}.txt", augmented_txt)

                else:
                    step(image_augmentation_instance)
                    
                    augmented_data = image_augmentation_instance.augmented_data
                    current_augment = image_augmentation_instance.current_augment
                    augmented_txt = image_augmentation_instance.augmented_txt

                    cv2.imwrite(f"{self.augmented_folder}/{data_row['Name']}__{current_augment}.png", augmented_data)
                    np.savetxt(f"{self.augmented_folder}/{data_row['Name']}__{current_augment}.txt", augmented_txt)

        return None

    # UTILS
    def _fileNaming(self, file):
        """
        Hidden method.
        Returns the absolute path, name, and extension of a given file.
        """

        file_path = self.data_folder + "/"
        file_name = file[:file.rindex(".")]
        file_extension = file[file.rindex(".")+1:]

        return file_path, file_name, file_extension
    
    def _parse_txt(self, txt_file):
        """
        Hidden method.
        Returns the labels and bbox positions of a txt file into the YOLO format.
        """
        # YOLO Datasets format
        file = pd.read_csv(f"{txt_file}.txt", delimiter=" ", header=None)
        labels = file.iloc[:, 0].values.tolist()
        locations = file.iloc[:, 1:5].values.tolist() 
        return labels, locations

    def _parse_xml(self, xml_file):
        """
        Hidden method.
        Returns the labels and bbox positions of a xml file into the YOLO format.
        """
        # PASCAL_VOC Datasets format
        tree = ET.parse(xml_file)
        root = tree.getroot()

        labels = []
        locations = []

        for obj in root.findall('.//object'):
            label = obj.find('name').text
            labels.append(label)

            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            x_center = (xmin + xmax)//2
            y_center = (ymin + ymax)//2
            width = xmax - xmin
            height = ymax - ymin

            locations.append([x_center, y_center, width, height])

        return labels, locations    

    def _parse_json(json_file):
        """
        Hidden method.
        Returns the labels and bbox positions of a json file into the YOLO format.
        """
        # COCO datasets format
        with open(json_file, 'r') as f:
            data = json.load(f)

        labels = []
        locations = []

        for annotation in data['annotations']:
            label = annotation['category_id']
            labels.append(label)

            bbox = annotation['bbox']
            xmin = bbox[0]
            ymin = bbox[1]
            width = bbox[2]
            height = bbox[3]

            x_center = xmin + width//2
            y_center = ymin + height//2

            locations.append([x_center, y_center, width, height])

        return labels, locations
    
    def _parse_tfrecord(tfrecord_file):
        """
        Hidden method.
        Returns the labels and bbox positions of a tfrecords file into the YOLO format.
        """
        # labels = []
        # bounding_boxes = []

        # for example in tf.data.TFRecordDataset([tfrecord_file]):
        #     tf_example = tf.train.Example.FromString(example.numpy())

        #     label = tf_example.features.feature['label'].int64_list.value[0]
        #     labels.append(label)

        #     bbox = tf_example.features.feature['bbox'].float_list.value
        #     bounding_boxes.append((bbox[0], bbox[1], bbox[2], bbox[3]))

        # return labels, bounding_boxes

    def _load_config(self):
        """
        Loads the config.ini file and loads all its parameters.
        """
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(DIR_DATA_CONFIGS_PATH, "AugmentationsConfig.ini"))
        return self.config
    
    def _read_config(self):
        """
        Retreives the parameters relevants to the pipelineController class.
        """
        self.data_type = self.config.get(self.run_config, "data_type")
        self.config_augmentation_list = self.config.get(self.run_config, "augmentation_list")

        return None
    
    def _reset_config(self):
        
        self.config["GUI"]["augmentation_list"] = "['gray_scale', 'rotation']"
        self.config["GUI"]["rotation_angle"] = "30"
        self.config["GUI"]["rotation_resize"] = "1"
        self.config["GUI"]["noise_std"] = "0.5"
        self.config["GUI"]["channels_swap_rgb_order"] = "2, 0, 1"
        self.config["GUI"]["brightness_coeff"] = "120"
        self.config["GUI"]["contrast_coeff"] = "120"
        self.config["GUI"]["zoom_coeff"] = "80"

        with open(os.path.join(DIR_DATA_CONFIGS_PATH, "AugmentationsConfig.ini"), 'w') as configfile:
            self.config.write(configfile)
        
        return None

    def _debug_config(self):
        """
        Prints the config.ini file content.
        """
        print(self.config.sections())
        for section in self.config.sections():
            print(f"[{section}]")
            for key, value in self.config.items(section):
                print(f"{key} = {value}")
        return None


import matplotlib.pyplot as plt
import pandas as pd

class Display():

    def __init__(self, image_path, bbox_path) -> None:
        
        self.image_path = image_path
        self.bbox_path = bbox_path

    # DISPLAY CALLS
    def display(self, ax, show=True):

        try:
            self.df = pd.read_csv(self.bbox_path, header=None, delimiter=' ')
            bbox = True
        except:
            bbox = False
            
        if bbox:
            bbox_pseudo_pipeline = [
                self._readImage(),
                self._readPositions(),
                self._plotBBoxImage(ax),
            ]
            for step in bbox_pseudo_pipeline:
                step

        else:
            simple_pseudo_pipeline = [
                self._readImage(),
                self._plotImage(ax),
            ]
            for step in simple_pseudo_pipeline:
                step

        if show:
            plt.show()

    def displayAndCompare(self):
        pass

    # UTILS
    def _readPositions(self):
        # df = pd.read_csv(self.labels_path, header=None, delimiter=' ')
        df = self.df
        df.columns = ['Target', 'center_x', 'center_y', 'width', 'height']

        # Centers are moved to the top left corners, as plt.Rectantgle uses it as reference
        df['bottom_x'] = ((df['center_x'] - df['width'] / 2) * self.width).astype(int)
        df['bottom_y'] = ((df['center_y'] - df['height'] / 2) * self.height).astype(int)
        df['width'] = (df['width'] * self.width).astype(int)
        df['height'] = (df['height'] * self.height).astype(int)

        self.df = df[["bottom_x", "bottom_y", "width", "height", "Target"]]
                
    def _readImage(self):
        self.image = plt.imread(self.image_path)
        self.height, self.width, self.channels = self.image.shape[0:3]

    def _plotBBoxImage(self, ax):

        # fig, ax = plt.subplots(1)
        ax.imshow(self.image)
        ax.set_title(f"Applied augmentation :\n{self.image_path[self.image_path.rindex('__')+2:-4]}")

        for row_idx, row in self.df.iterrows():
            x, y, width, height, target = row
            rect = plt.Rectangle((x, y), width, height, linewidth=2, edgecolor='g', facecolor='none')
            ax.scatter(x, y, color="r")
            ax.add_patch(rect)
            ax.text(x, y - 5, f'Class {target}', color='crimson', fontsize=8)

        # ax.set_xticks([0, self.image.shape[0]])
        # ax.set_yticks([0, self.image.shape[1]])   

    def _plotImage(self, ax):
        
        # fig, ax = plt.subplots(1)
        ax.imshow(self.image)
        ax.set_title(f"Applied augmentation :\n{self.image_path[self.image_path.rindex('__')+2:-4]}")


