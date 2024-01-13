import cv2
import numpy as np
import random
import configparser

class ImageAugmentation():
    """
    Class of methods for image augmentation 
    """
    def __init__(self, data, labels, positions, config, run_config="DEFAULT") -> None:
        """
        Attributes:
            augmented_data: The augmented datapoint (image).
            augmented_labels: The augmented label(s). Updated if a feature is considered lost due to an augmentation.
            augmented_positions: The augmented boudary boxe(s), to keep track of the features after the augmentation.  
            current_augment: The name of the applied augmentation.

        Args:
            data: The datapoint to augment, expects an array-like object of shape [width, height, channels] or [width, height].
            labels : The data label(s), expects an array-like object of shape [n]
            positions : The data boundary boxe(s) values, expects an array-like object of shape [n, 4]
        """

        self.data = data
        self.labels = labels
        self.positions = positions
        self.config = config
        self.run_config = run_config

        self.augmented_data = None
        self.augmented_labels = None
        self.augmented_positions = None
        self.augmented_txt = None
        self.current_augment = ""

        # Ensures the data is 3D by promoting it (Inputs should always be 2D or 3D)
        if len(self.data.shape) <= 2:
            self.data.reshape(self.data.shape + (1,))
        self.height, self.width, self.channels = self.data.shape[0:3]

    def _load_config(self):
        # Config attributes are read by the classes 
        config = configparser.ConfigParser()
        config.read(self.main)
        return config

    # CONSERVATIVE TRANSFORMATIONS
    def imageGrayScale(self):
        """
        Converts the image to a gray level image of three identical channels.
        """
        self.current_augment = "gray_scale"

        self.augmented_data = np.stack([cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)] * 3, axis=-1)

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data

    def imageChannelsSwap(self):
        """
        Swaps the RGB channels into a random organization. Can't return the same RGB order image.
        Default switches RGB to BRG.
        Does nothing to gray level images.
        """
        self.current_augment = "channels_swap"

        rgb_order = [0, 1, 2]
        rgb_permutations = ["rgb", "rbg", "grb", "gbr", "brg", "bgr"]

        config_std = self.config.get(self.run_config, "channels_swap_rgb_order").lower()
        if config_std == "random":
            random.shuffle(rgb_order)
        elif config_std in rgb_permutations:
            rgb_order = (config_std.index("r")%3, config_std.index("g")%3, config_std.index("b")%3)
        else:
            rgb_order = [2, 0, 1]
        self.augmented_data = self.data[:, :, tuple(rgb_order)]

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data

    def imageNoise(self):
        """
        Adds a layer of gaussian (white) noise to an image. 
        Default: µ=0, std=1
        """
        self.current_augment = "noise"

        config_std = self.config.get(self.run_config, "noise_std")
        if config_std == "random":
            std = abs(np.random.rand() + np.random.rand())
        elif config_std != '':
            std = float(config_std)
        else:
            std = 1

        noise = np.random.normal(loc=0, scale=std, size=self.data.shape).astype('uint8')
        self.augmented_data = cv2.add(self.data, noise)

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data
    
    def imageZoom(self):
        # safe image crop output and/or logic could be used as a CutMix input

        self.current_augment = "zoom"
        self.augmented_data = self.data

        config_std = self.config.get(self.run_config, "zoom_coeff")
        if config_std == "random":
            zoom_coeff = abs(np.random.rand() + np.random.rand())
        elif config_std != '':
            zoom_coeff = min(float(config_std)/100, 5)
        else:
            zoom_coeff = 1

        zoom_matrix = np.array([[zoom_coeff, 0, 0], [0, zoom_coeff, 0]], dtype=np.float32)

        dsize = (self.data.shape[1], self.data.shape[0])
        # dsize = (int(self.width * zoom_coeff), int(self.height * zoom_coeff))
        self.augmented_data = cv2.warpAffine(self.data, zoom_matrix, dsize=dsize)

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data
    
    def imageCrop(self):
        
        self.current_augment = "crop"
        self.augmented_data = self.data

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data

    def imageCut(self):

        self.current_augment = "cut"

        left = np.random.randint(low=0, high=self.width)
        right = np.random.randint(low=left, high=self.width)
        bottom = np.random.randint(low=0, high=self.height)
        top = np.random.randint(low=bottom, high=self.height)

        self.augmented_data = self.data.copy()
        self.augmented_data[bottom:top, left:right, :] = np.zeros((top-bottom, right-left, self.channels))

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data

    def imageBrightness(self):

        self.current_augment = "brightness"

        config_std = self.config.get(self.run_config, "brightness_coeff")
        if config_std == "random":
            brightness_factor = abs(np.random.randn()*2)
        elif config_std != '':
            brightness_factor = float(config_std)/100
        else:
            brightness_factor = 1.5

        self.augmented_data = cv2.addWeighted(self.data, brightness_factor, self.data, 0, 0)

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data
    
    def imageColourInversion(self):

        self.current_augment = "colour_inversion"

        self.augmented_data = 255 - self.data

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data

    def imageContrast(self):
        pass
        self.current_augment = "contrast"

        config_contrast = self.config.get(self.run_config, "contrast_coeff")
        if config_contrast == "random":
            contrast_coeff = np.random.randint(low=1, high=20)/10
        elif config_contrast != '':
            contrast_coeff = float(config_contrast)/100
        else:
            contrast_coeff = 0.8
        self.augmented_data = np.power(self.data / 255.0, contrast_coeff) * 255.0
        self.augmented_data = self.augmented_data.astype(np.uint8)

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data

    def imageBlur(self):
        """
        Applies a gaussian blur to the input data
        """
        self.current_augment = "blur"

        self.augmented_data = cv2.GaussianBlur(self.data, ksize=(3, 3), sigmaX=0, sigmaY=0)

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data

    def imageEdges(self):
        """
        Standard Sobel's kernel for edges detection.
        """
        data = self.imageGrayScale()
        self.current_augment = "edges"

        sobel_x = cv2.Sobel(data, cv2.CV_64F, dx=1, dy=0, ksize=3)
        sobel_y = cv2.Sobel(data, cv2.CV_64F, dx=0, dy=1, ksize=3)

        self.augmented_data = np.sqrt(sobel_x**2 + sobel_y**2)

        self.augmented_txt = []
        if self.positions != None:
            for idx, target in enumerate(self.positions):
                self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
                x_center, y_center, width, height = target # retreives the positions of the bbox
                self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
                self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_data



    # NON CONSERVATIVE TRANSFORMATIONS
    def imageHorizontalFlip(self):
        """
        Flips an image horizontally.
        """
        self.current_augment = "horizontal_flip"
        self.augmented_data = cv2.flip(self.data, 1)
        return self.augmented_data

    def labelsHorizontalFlip(self):
        """
        Edits the flipped boundary boxes by flipping the X-axis positions.
        Should be called immediatly after imageHorizontalFlip().
        """

        self.augmented_txt = []
        for idx, bbox in enumerate(self.positions):
            self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
            x_center, y_center, width, height = bbox # retreives the positions of the bbox
            self.augmented_positions = [1-x_center, y_center, width, height] # rearranges the positions accordingly
            self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]
        return self.augmented_txt

    def imageVerticalFlip(self):
        """
        Flips an image vertically.
        """
        self.current_augment = "vertical_flip"
        self.augmented_data = cv2.flip(self.data, 0)
        return self.augmented_data

    def labelsVerticalFlip(self):
        """
        Edits the flipped boundary boxes by flipping the Y-axis positions.
        Should be called immediatly after imageVerticalFlip().
        """

        self.augmented_txt = []
        for idx, target in enumerate(self.positions):
            self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
            x_center, y_center, width, height = target # retreives the positions of the bbox
            self.augmented_positions = [x_center, 1-y_center, width, height] # rearranges the positions accordingly
            self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]
        return self.augmented_txt
    
    def imageZoom(self):
        """
        Zooms/Dezooms the image to a certain size while keeping the same frame dimensions.
        """
        self.current_augment = "zoom"

        config_zoom = self.config.get(self.run_config, "zoom_coeff")
        if config_zoom == "random":
            zoom_factor = np.random.randint(low=1, high=20)/10
        elif config_zoom != '':
            zoom_factor = float(config_zoom)/100
        else:
            zoom_factor = 0.8

        config_zoom_direction = self.config.get(self.run_config, "zoom_coeff")
        if config_zoom_direction in ["top_left", "top_right", "bottom_left", "bottom_right", "center"]:
            zoom_direction = config_zoom_direction
        else:
            zoom_direction = "center"

        return self._imageZoom(zoom_factor=zoom_factor, zoom_type=None, zoom_direction=zoom_direction)

    def _imageZoom(self, delta_x=0, delta_y=0, zoom_factor=0.8, zoom_type=None, zoom_direction="center"):
        """
        Zooms/Dezooms the image to a certain size while keeping the same resolution.
        May be called by other transformations as utils (rotation) or under the hood transformation (translation, crop...)
        """

        # Creates an unique attribute to be passed onto labelsZoom if necessary
        self.zoom_factor = zoom_factor

        if zoom_type == "crop":
            None
        elif zoom_type == "translation":
            zoom_center_x = self.width//2 + delta_x 
            zoom_center_y = self.height//2 + delta_y
        elif zoom_type == "rotation":
            zoom_center_x = self.width//2 
            zoom_center_y = self.height//2
        else:
            if zoom_direction == "top_left":
                zoom_center_x = self.width//2 - int(self.width * (1 - zoom_factor))
                zoom_center_y = self.height//2 + int(self.height * (1 - zoom_factor))
            elif zoom_direction == "top_right":
                zoom_center_x = self.width//2 + int(self.width * (1 - zoom_factor))
                zoom_center_y = self.height//2 + int(self.height * (1 - zoom_factor))
            elif zoom_direction == "bottom_left":
                zoom_center_x = self.width//2 - float(self.width * (1 - zoom_factor))
                zoom_center_y = self.height//2 - float(self.height * (1 - zoom_factor))
            elif zoom_direction == "bottom_right":
                zoom_center_x = self.width//2 + int(self.width * (1 - zoom_factor))
                zoom_center_y = self.height//2 - int(self.height * (1 - zoom_factor))
            else:
                zoom_direction = "center"
                zoom_center_x = self.width//2 
                zoom_center_y = self.height//2

        new_height = int(self.height * zoom_factor)
        new_width = int(self.width * zoom_factor)

        # Image is zoomed/translated towards this point coordinates:
        translation_x = zoom_center_x - (new_width // 2)
        translation_y = zoom_center_y - (new_height // 2)
        # Faked rotation matrix, with only translation components and an identity whose gain is the zoom_factor
        translation_matrix = np.float32([[zoom_factor, 0, translation_x], [0, zoom_factor, translation_y]])

        self.augmented_data = cv2.warpAffine(self.data, translation_matrix, (self.width, self.height), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        return self.augmented_data
    
    def labelsZoom(self):
        """
        Edits the zoomed boundary boxes by mooving their centers and ratios.
        Should be called immediatly after imageZoom().
        """

        self.augmented_txt = []
        for idx, target in enumerate(self.positions):
            self.augmented_labels = [self.labels[idx]] # retreives the label/target (1st column of the txt)
            x_center, y_center, width, height = target # retreives the positions of the bbox

            x_vector = x_center - 0.5
            y_vector = y_center - 0.5

            x_center = 0.5 + x_vector * self.zoom_factor
            y_center = 0.5 + y_vector * self.zoom_factor

            width = width*self.zoom_factor
            height = height*self.zoom_factor
            
            self.augmented_positions = [x_center, y_center, width, height] # rearranges the positions accordingly
            self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_txt      
    
    def imageTranslation(self):
        self.current_augment = "translation"

        delta_x = 100
        delta_y = 0
        self.imageZoom(self.current_augment, delta_x, delta_y)

        return self.augmented_data
    
    def labelsTranslation(self):

        self.labelsZoom(self.current_augment)

        return self.augmented_txt

    def imageRotation(self):
        """
        Rotates an image randomly.
        """
        self.current_augment = "rotation"

        config_angle = self.config.get(self.run_config, "rotation_angle")
        if config_angle == "random":
            self.angle = np.round(np.random.rand() * 180)
        elif config_angle != '':
            self.angle = float(config_angle)
        else:
            self.angle = 30

        # If resize, the image is unzoomed so that the rotated image fits in a same size frame, without being cropped.
        # Else the image is cropped to return an augmented image of the same size.
        config_rezise = self.config.get(self.run_config, "rotation_resize")
        if config_rezise != "0":
            angle_rad = np.radians(abs(self.angle)%180)
            zoom_factor = 1
            if (0 <= angle_rad < np.pi/4):
                angle_rad = angle_rad

            elif (np.pi/4 <= angle_rad < np.pi/2):
                angle_rad = angle_rad

            elif (np.pi/2 <= angle_rad < 3*np.pi/4):
                angle_rad = angle_rad - np.pi/4

            elif (3*np.pi/4 <= angle_rad <= np.pi):
                angle_rad = abs(angle_rad - np.pi)
            
            denominator = abs(self.width * np.sin(angle_rad) + self.height * np.cos(angle_rad))
            zoom_factor = self.height / denominator

            data = self._imageZoom(zoom_type=self.current_augment, zoom_factor=zoom_factor)
            self.labelsZoom()
        else:
            data = self.data
        
        center = tuple(np.array([self.width, self.height]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.angle, scale=1)
        dsize = (self.data.shape[1], self.data.shape[0])

        self.augmented_data = cv2.warpAffine(data, rotation_matrix, dsize=dsize, flags=cv2.INTER_LINEAR)

        return self.augmented_data
    
    def labelsRotation(self):
        """
        Edits the boundary boxes to match the rotation.
        Should be called immediatly after imageRotation().
        """

        center = tuple(np.array([self.width, self.height]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.angle, 1.0)

        # If the image was resized to avoid cropping
        config_rezise = self.config.get(self.run_config, "rotation_resize")
        if config_rezise != "0":
            positions = np.array(self.augmented_txt)[:, 1:]
        else:
            positions = self.positions

        self.augmented_positions = []
        self.augmented_txt = []
        for bbox_idx, bbox in enumerate(positions):
            self.augmented_labels = [self.labels[bbox_idx]]

            [x, width] = [bbox[0] * self.width, bbox[2] * self.width]
            [y, height] = [bbox[1] * self.height, bbox[3] * self.height]
            
            # All the rotations are computed in the image space (eg: 1920x1080, and not 1.00x1.00)
            center = np.array([x, y])
            rotated_center = rotation_matrix[:, 0:2] @ center + rotation_matrix[:, 2]

            top_left = np.array([x - width/2, y + height/2])
            rotated_top_left = rotation_matrix[:, 0:2] @ top_left + rotation_matrix[:, 2]

            bottom_right = np.array([x + width/2, y - height/2])
            rotated_bottom_right = rotation_matrix[:, 0:2] @ bottom_right + rotation_matrix[:, 2]

            bottom_left = np.array([x - width/2, y - height/2])
            rotated_bottom_left = rotation_matrix[:, 0:2] @ bottom_left + rotation_matrix[:, 2]

            top_right = np.array([x + width/2, y + height/2])
            rotated_top_right = rotation_matrix[:, 0:2] @ top_right + rotation_matrix[:, 2]

            new_width = abs(np.max([rotated_top_left[0], rotated_bottom_left[0], rotated_top_right[0], rotated_bottom_right[0]]) - np.min([rotated_top_left[0], rotated_bottom_left[0], rotated_top_right[0], rotated_bottom_right[0]]))
            new_height = abs(np.max([rotated_top_left[1], rotated_bottom_left[1], rotated_top_right[1], rotated_bottom_right[1]]) - np.min([rotated_top_left[1], rotated_bottom_left[1], rotated_top_right[1], rotated_bottom_right[1]]))

            # Rotations are stored in YOLO format, ie pixel_size/total_size 
            self.augmented_positions = [rotated_center[0]/self.width, rotated_center[1]/self.height, new_width/self.width, new_height/self.height]
            self.augmented_txt = self.augmented_txt + [self.augmented_labels + self.augmented_positions] # [ [label/target, positions], ...]

        return self.augmented_txt



