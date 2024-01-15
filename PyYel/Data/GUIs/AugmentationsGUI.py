import tkinter as tk
from tkinter import ttk
import configparser
import os

class ConfigApp(tk.Tk):
    def __init__(self, config_path):
        super().__init__()

        self.title("Config editor")
        self.config_path = config_path

        # Suptitle
        text = "\tConfigure the augmentations, empty parameters will be replaced with default values.\t"
        self.label_description = ttk.Label(self, text=text)
        self.label_description.grid(row=0, column=0, columnspan=4, pady=10)

        # Argument values
        self.label_editor = ttk.Label(self, text="Enter augmentation parameters ([random]/[val]):")
        self.label_editor.grid(row=1, column=0, columnspan=2, pady=10)
        self.entries = {}
        self.entries["rotation_angle"] = self.create_entry("Rotation angle (°):", row=2)
        self.entries["noise_std"] = self.create_entry("Gaussian noise std:", row=3)
        self.entries["channels_swap_rgb_order"] = self.create_entry("New RGB order (rgb/bgr/gbr...):", row=5)
        self.entries["brightness_coeff"] = self.create_entry("Brightness coefficient (%):", row=7)
        self.entries["contrast_coeff"] = self.create_entry("Contrast coefficient (%):", row=8)
        self.entries["zoom_coeff"] = self.create_entry("Zoom coefficient (%):", row=10)

        # Augmentation selection
        self.label_selector = ttk.Label(self, text="Select augmentation methods:")
        self.label_selector.grid(row=1, column=2, columnspan=4, pady=10)
        self.checkbox_vars = {}
        self.create_checkbox("rotation", row=2)
        self.create_checkbox("noise", row=3)
        self.create_checkbox("gray_scale", row=4)
        self.create_checkbox("channels_swap", row=5)
        self.create_checkbox("cut", row=6)
        self.create_checkbox("brightness", row=7)
        self.create_checkbox("contrast", row=8)
        self.create_checkbox("colour_inversion", row=9)
        self.create_checkbox("zoom", row=10)
        self.create_checkbox("horizontal_flip", row=11)
        self.create_checkbox("vertical_flip", row=12)
        self.create_checkbox("blur", row=13)
        self.create_checkbox("edges", row=14)


        self.save_button = ttk.Button(self, text="Save Config", command=self.save_config)
        self.save_button.grid(row=15, column=0, columnspan=4, pady=10)

    def create_entry(self, label_text, row):
        ttk.Label(self, text=label_text).grid(row=row, column=0, padx=5)
        entry = ttk.Entry(self)
        entry.grid(row=row, column=1, padx=0)
        return entry

    def create_checkbox(self, label_text, row):
        var = tk.IntVar()
        checkbox = ttk.Checkbutton(self, text=label_text, variable=var)
        checkbox.grid(row=row, column=2, padx=5, sticky="w")
        self.checkbox_vars[label_text] = var

    def save_config(self):
        config = configparser.ConfigParser()
        config.read(self.config_path)

        # Save augmentation settings
        for key, entry in self.entries.items():
            value = entry.get()
            config["CUSTOM"][key] = value

        # Save augmentation methods
        selected_options = [label for label, var in self.checkbox_vars.items() if var.get() == 1]
        config["CUSTOM"]["augmentation_list"] = f"{selected_options}"

        with open(self.config_path, 'w') as configfile:
            config.write(configfile)

        return self.destroy()

if __name__ == "__main__":
    config_path = os.path.abspath("") + "/config.ini"
    app = ConfigApp(config_path=config_path)
    app.mainloop()

