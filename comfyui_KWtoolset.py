import numpy as np
import torch
import os
import sys
import folder_paths
import comfy.utils
import node_helpers
import comfy.sd
from PIL import Image, ImageOps, ImageSequence, ImageFile
from .sd_prompt_reader.__version__ import VERSION as CORE_VERSION
from .sd_prompt_reader.constants import SUPPORTED_FORMATS
from .sd_prompt_reader.image_data_reader import ImageDataReader
import re
import cv2
import logging

from PIL import Image
from matplotlib import cm
from .open_pose.util import draw_bodypose, draw_handpose, draw_facepose, HWC3, resize_image
import copy  # 用于深拷贝

#from controlnet_aux.open_pose.util import draw_bodypose, draw_handpose, draw_facepose, HWC3, resize_image

#from controlnet_aux.open_pose import PoseResult

#from .open_pose import OpenposeDetector, PoseResult
#from .open_pose.util import draw_bodypose, draw_handpose, draw_facepose

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def register_node(identifier: str, display_name: str):
    def decorator(cls):
        NODE_CLASS_MAPPINGS[identifier] = cls
        NODE_DISPLAY_NAME_MAPPINGS[identifier] = display_name
        return cls
    return decorator

BLUE = "\033[1;34m"
CYAN = "\033[36m"
RESET = "\033[0m"

ERROR_MESSAGE = {
    "format_error": "No data detected or unsupported format. "
    "Please see the README for more details.\n"
    "https://github.com/",
    "complex_workflow": "The workflow is overly complex, or unsupported custom nodes have been used. "
    "Please see the README for more details.\n"
    "https://github.com/",
}

def output_to_terminal(text: str):
    print(f"{RESET+BLUE}" f"[SD Prompt Reader] " f"{CYAN+text+RESET}")



@register_node("KwtoolsetLoraLoaderwithpreview", "KW Lora Loader with preview")
class KwtoolsetLoraLoaderwithpreview:
    CATEGORY = "Kwtoolset"
    INPUT_TYPES = lambda: {
        "required": { 
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
        }
    }
    RETURN_NAMES = ("MODEL", "CLIP", "IMAGE", "MASK", "Lora positive", "Lora negative", "Lora setting")
    RETURN_TYPES = ("MODEL", "CLIP", "IMAGE", "IMAGE", "STRING", "STRING", "STRING")
    #OUTPUT_IS_LIST = (False, False, False, False, True, True, True)
    FUNCTION = "load_lora"

    def __init__(self):
        self.loaded_lora = None

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, None, None, "", "", "")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        image_path = lora_path.replace(".safetensors", ".preview.png")

        script_dir = os.path.dirname(__file__)
        default_image_path = os.path.join(script_dir, 'default.png')
        
        if os.path.exists(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_image, output_mask = self.load_image(image_path)
            positive, negative, setting = self.read_image_data(image_path)
        else:
            output_image, output_mask = self.load_image(default_image_path)
            positive, negative, setting = self.read_image_data(default_image_path)
            

        image_path_list = [image_path] if os.path.exists(image_path) else []

        return (model_lora, clip_lora, output_image, output_mask, positive, negative, setting)

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            return None, None
            
        try:
            img = node_helpers.pillow(Image.open, image_path)
        except Exception as e:
            #print(f"Error opening image: {e}")
            return None, None
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    def read_image_data(self, image_path):
    
        if not os.path.exists(image_path):
            return None, None, None
            
        try:
            with open(image_path, "rb") as f:
                image_data = ImageDataReader(f)
                if image_data.status.name == "COMFYUI_ERROR":
                    output_to_terminal(ERROR_MESSAGE["complex_workflow"])
                    return "", "", ""
                elif image_data.status.name in ["FORMAT_ERROR", "UNREAD"]:
                    output_to_terminal(ERROR_MESSAGE["format_error"])
                    return "", "", ""

                positive = image_data.positive   
                negative = image_data.negative
                setting = image_data.setting

                #positive =''.join(image_data.positive)
                #negative =''.join(image_data.negative)
                #setting =''.join(image_data.setting)
                
                output_to_terminal("Positive: \n" + positive)
                output_to_terminal("Negative: \n" + negative)
                output_to_terminal("Setting: \n" + setting)
                
        except Exception as e:
            #print(f"Error reading image data: {e}")
            return "", "", ""         

        return positive, negative, setting


@register_node("KwtoolsetCheckpointLoaderwithpreview", "KW Checkpoint Loader with preview")
class KwtoolsetCheckpointLoaderwithpreview:
    CATEGORY = "Kwtoolset"
    INPUT_TYPES = lambda: {
        "required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                    }
    }
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "NAME_STRING","IMAGE", "ckpt positive", "ckpt negative", "ckpt setting", "Seed")

    #OUTPUT_IS_LIST = (False, False, False, False, True, True, True)
    FUNCTION = "load_checkpoint"


    def __init__(self):
        self.loaded_lora = None

    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        
        image_path = ckpt_path.replace(".safetensors", ".preview.png")

        script_dir = os.path.dirname(__file__)
        default_image_path = os.path.join(script_dir, 'default.png')
        
        if os.path.exists(image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_image, output_mask = self.load_image(image_path)
            positive, negative, setting, seed = self.read_image_data(image_path)
        else:
            output_image, output_mask = self.load_image(default_image_path)
            positive, negative, setting, seed = self.read_image_data(default_image_path)
            
        #image_path_list = [image_path] if os.path.exists(image_path) else []
        
        return (out[0], out[1], out[2], os.path.splitext(os.path.basename(ckpt_name))[0],output_image, positive, negative, setting, seed)

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            return None, None

        try:
            img = node_helpers.pillow(Image.open, image_path)
        except Exception as e:
            #print(f"Error opening image: {e}")
            return None, None
            
     
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    def read_image_data(self, image_path):
    
        if not os.path.exists(image_path):
            return "", "", "",-1
            
        try:
            with open(image_path, "rb") as f:
                image_data = ImageDataReader(f)
                if image_data.status.name == "COMFYUI_ERROR":
                    output_to_terminal(ERROR_MESSAGE["complex_workflow"])
                    return "", "", "",-1
                elif image_data.status.name in ["FORMAT_ERROR", "UNREAD"]:
                    output_to_terminal(ERROR_MESSAGE["format_error"])
                    return "", "", "",-1

                positive = image_data.positive   
                negative = image_data.negative
                setting = image_data.setting
                #seed = '234'

                #positive =''.join(image_data.positive)
                #negative =''.join(image_data.negative)
                #setting =''.join(image_data.setting)
                
                output_to_terminal("Positive: \n" + positive)
                output_to_terminal("Negative: \n" + negative)
                output_to_terminal("Setting: \n" + setting)
                #output_to_terminal("Seed: \n" + seed)
                
        except Exception as e:
            #print(f"Error reading image data: {e}")
            return "", "", "", -1

        if len(setting) > 0:
            seed = self.extract_seed(setting)
        else:
            seed = -1
        
        return positive, negative, setting , seed

    def extract_seed(self, data: str):
        try:
            match = re.search(r"Seed:\s*(\d+)", data, re.IGNORECASE)
            if match:
                return match.group(1)
            else:
                return 0
        except Exception as e:
            print(f"Error extracting seed: {e}")
            return 0
            


@register_node("KwtoolsetLoadCheckpointsBatch", "KW Load Checkpoints Batch")
class KwtoolsetLoadCheckpointsBatch:
    INPUT_TYPES = lambda: {
        "required": {
            "directory": ("STRING", {"default": ''}),
            "CheckpointMainDirectory": ("STRING", {"default": ''}),
            "index": ("INT", {"default": 0, "min": 0, "max": 150000, "step": 1}),
        },
        "optional": {
            "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
        }
    }
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("PathName", "Name","NameWithPath")
    #OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_checkpoints"
    CATEGORY = "Kwtoolset"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if 'load_always' in kwargs and kwargs['load_always']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs.items()))

    def load_checkpoints(self, directory: str,CheckpointMainDirectory: str, index: int = 0, load_always=False):
        ckpt_path = folder_paths.folder_names_and_paths["checkpoints"]
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        
        dir_files = os.listdir(directory)
        if len(dir_files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        # Filter files by extension
        valid_extensions = ['.ckpt', '.safetensors']
        dir_files = [f for f in dir_files if any(f.lower().endswith(ext) for ext in valid_extensions)]

        dir_files = sorted(dir_files)

        if index >= len(dir_files) or index < 0:
            raise IndexError(f"Index {index} is out of range for available checkpoints.")

        checkpoint_path = os.path.join(directory, dir_files[index])
        checkpoint_name = os.path.basename(checkpoint_path)
        checkpoint_name_with = checkpoint_path.replace(CheckpointMainDirectory, "")
        
        #checkpoint_paths = [os.path.join(directory, f) for f in dir_files]
        #checkpoint_names = [os.path.basename(f) for f in checkpoint_paths]

        return checkpoint_path, checkpoint_name,checkpoint_name_with
   
   
from nodes import MAX_RESOLUTION
import scipy.ndimage
@register_node("KwtoolsetGrowMaskPlus", "KW Grow Mask Plus")
class KwtoolsetGrowMaskPlus:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "Movex": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "Movey": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "Extendx": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "Extendy": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                #"tapered_corners": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "mask"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "expand_mask"
        
    def expand_mask(self, mask, Movex, Movey, Extendx, Extendy):
        #c = 0 if tapered_corners else 1

        mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        out = []

        for m in mask:
            output = m.numpy()
            for _ in range(abs(Movex)):
                if Movex > 0:
                    kernel = np.array([[0, 0, 0],
                                       [0, 0, 1],
                                       [0, 0, 0]])
                else:
                    kernel = np.array([[0, 0, 0],
                                       [1, 0, 0],
                                       [0, 0, 0]])
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            for _ in range(abs(Movey)):
                if Movey > 0:
                    kernel = np.array([[0, 1, 0],
                                       [0, 0, 0],
                                       [0, 0, 0]])
                else:
                    kernel = np.array([[0, 0, 0],
                                       [0, 0, 0],
                                       [0, 1, 0]])
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            
            for _ in range(abs(Extendx)):
                if Extendx > 0:
                    kernel = np.array([[0, 0, 0],
                                       [0, 1, 1],
                                       [0, 0, 0]])
                else:
                    kernel = np.array([[0, 0, 0],
                                       [1, 1, 0],
                                       [0, 0, 0]])
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            for _ in range(abs(Extendy)):
                if Extendy > 0:
                    kernel = np.array([[0, 1, 0],
                                       [0, 1, 0],
                                       [0, 0, 0]])
                else:
                    kernel = np.array([[0, 0, 0],
                                       [0, 1, 0],
                                       [0, 1, 0]])
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            output = torch.from_numpy(output)
            out.append(output)
        return (torch.stack(out, dim=0),)


@register_node("KwtoolsetGetHipMask", "KW Get Hip Mask")
class KwtoolsetGetHipMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "image_width": ("INT", {"min": 0, "max": MAX_RESOLUTION}),
                "image_height": ("INT", {"min": 0, "max": MAX_RESOLUTION}),
            },
            "optional": {
                "person_number": ("INT", {"default": 0}),
                "points_list": ("STRING", {"multiline": True, "default": "8,11,9,12"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "MASK")
    RETURN_NAMES = ("x", "y", "width", "height", "mask")
    FUNCTION = "box_keypoints"
    CATEGORY = "utils"

    def get_keypoint_from_list(self, keypoint_list, item):
        idx_x = item * 3
        idx_y = idx_x + 1
        idx_conf = idx_y + 1
        return keypoint_list[idx_x], keypoint_list[idx_y], keypoint_list[idx_conf]

    def box_keypoints(self, pose_keypoint, image_width, image_height, person_number=0, points_list='8,11,9,12'):
        points_we_want = [int(element) for element in points_list.split(",")]

        min_x = MAX_RESOLUTION
        min_y = MAX_RESOLUTION
        max_x = 0
        max_y = 0

        people = pose_keypoint[0].get("people", [])
        if person_number >= len(people):
            return 0, 0, 0, 0, self.solid(0, image_width, image_height)[0]  # Handle case where person_number is out of range

        keypoints_exist = True
        if set(points_we_want) == {8, 9, 11, 12}:
            keypoints_found = []
            for element in points_we_want:
                x, y, z = self.get_keypoint_from_list(people[person_number]["pose_keypoints_2d"], element)
                if z > 0:
                    keypoints_found.append((element, x, y))

            if not keypoints_found:
                return 0, 0, 0, 0, self.solid(0, image_width, image_height)[0]

            if len(keypoints_found) == 1:
                element, x, y = keypoints_found[0]
                if element == 8:
                    min_x = x
                    max_x = x + x / 2
                    min_y = y
                    max_y = y + y / 5
                elif element == 11:
                    min_x = x - x / 3
                    max_x = x
                    min_y = y
                    max_y = y + y / 5
                elif element == 9:
                    min_x = x
                    max_x = x + x / 2
                    min_y = y - y / 4
                    max_y = y - y / 3
                elif element == 12:
                    min_x = x - x / 2
                    max_x = x
                    min_y = y - y / 4
                    max_y = y - y / 3
            elif len(keypoints_found) == 2 and {keypoints_found[0][0], keypoints_found[1][0]} == {8, 11}:
                x1, y1 = keypoints_found[0][1], keypoints_found[0][2]
                x2, y2 = keypoints_found[1][1], keypoints_found[1][2]
                min_x = min(x1, x2)
                max_x = max(x1, x2)
                min_y = min(y1, y2)
                max_y = max(y1, y2) +  (max(y1, y2) / 5) # Extend down by 50 pixels
                if max_y>1:
                    max_y = 1 
            else:
                for _, x, y in keypoints_found:
                    if x < min_x:
                        min_x = x
                    if y < min_y:
                        min_y = y
                    if x > max_x:
                        max_x = x
                    if y > max_y:
                        max_y = y
        else:
            for element in points_we_want:
                x, y, z = self.get_keypoint_from_list(people[person_number]["pose_keypoints_2d"], element)
                if z == 0:
                    continue
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

        min_x_int = int(min_x * image_width)
        min_y_int = int(min_y * image_height)
        width_int = int((max_x - min_x) * image_width)
        height_int = int((max_y - min_y) * image_height)

        # Create mask using solid
        mask = self.solid(1, width_int, height_int)[0]
        full_mask = self.solid(0, image_width, image_height)[0]
        full_mask[:, min_y_int:min_y_int + height_int, min_x_int:min_x_int + width_int] = mask

        return min_x_int, min_y_int, width_int, height_int, full_mask

    def solid(self, value, width, height):
        out = torch.full((1, height, width), value, dtype=torch.float32, device="cpu")
        return (out,)
        
        
@register_node("KwtoolsetGetHipMasktest", "KW Get Hip Mask-test")
class KwtoolsetGetHipMasktest:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "image_width": ("INT", {"min": 0, "max": MAX_RESOLUTION}),
                "image_height": ("INT", {"min": 0, "max": MAX_RESOLUTION}),
            },
            "optional": {
                "person_number": ("INT", {"default": 0}),
                "points_list": ("STRING", {"multiline": True, "default": "8,11,9,12"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "MASK")
    RETURN_NAMES = ("x", "y", "width", "height", "mask")
    FUNCTION = "box_keypoints"
    CATEGORY = "utils"

    def get_keypoint_from_list(self, list, item):
        idx_x = item * 3
        idx_y = idx_x + 1
        idx_conf = idx_y + 1
        return list[idx_x], list[idx_y], list[idx_conf]

    def box_keypoints(self, pose_keypoint, image_width, image_height, person_number=0, points_list='8,11,9,12'):
        points_we_want = [int(element) for element in points_list.split(",")]

        min_x = MAX_RESOLUTION
        min_y = MAX_RESOLUTION
        max_x = 0
        max_y = 0

        people = pose_keypoint[0].get("people", [])
        if person_number >= len(people):
            return 0, 0, 0, 0, self.solid(0, image_width, image_height)[0]  # Handle case where person_number is out of range

        for element in points_we_want:
            x, y, z = self.get_keypoint_from_list(people[person_number]["pose_keypoints_2d"], element)
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        min_x_int = int(min_x * image_width)
        min_y_int = int(min_y * image_height)
        width_int = int((max_x - min_x) * image_width)
        height_int = int((max_y - min_y) * image_height)

        # Create mask using solid
        mask = self.solid(1, width_int, height_int)[0]
        full_mask = self.solid(0, image_width, image_height)[0]
        full_mask[:, min_y_int:min_y_int+height_int, min_x_int:min_x_int+width_int] = mask

        return min_x_int, min_y_int, width_int, height_int, full_mask

    def solid(self, value, width, height):
        out = torch.full((1, height, width), value, dtype=torch.float32, device="cpu")
        return (out,)


@register_node("KwtoolsetGetImageSize", "KW Get Image Size")
class GetImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT","IMAGE")
    RETURN_NAMES = ("width", "height","image")
    FUNCTION = "execute"
    CATEGORY = "image"

    def execute(self, image):
        return (image.shape[2], image.shape[1],image)
        
        
@register_node("KWPositiveString", "KW Positive String")       
class KWPositiveString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "string_to_text"
    CATEGORY = "TEXT"
    def string_to_text(self, string):
        return (string, )

@register_node("KWNagetiveString", "KW Nagetive String")       
class KWNagetiveString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "string_to_text"
    CATEGORY = "TEXT"
    def string_to_text(self, string):
        return (string, )

       
@register_node("KWanywhereString", "KW anywhere String")       
class KWanywhereString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "string_to_text"
    CATEGORY = "TEXT"
    def string_to_text(self, string):
        return (string, )












def draw_pose(pose, H, W, draw_body=True, draw_hand=True, draw_face=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        # 准备调试信息
        debug_info = []
        debug_info.append("draw_hand_true" )
        debug_info.append("pose_keypoints:\n" + str(hands))
        # 将调试信息写入文件
        with open("debug_info_function.txt", "w") as f:
            f.write("\n\n".join(debug_info))
        
        canvas = draw_handpose(canvas, hands)

    if draw_face:
        canvas = draw_facepose(canvas, faces)

    return canvas









@register_node("KwtoolsetChangeOpenpose", "KW Change Openpose")
class KwtoolsetChangeOpenpose:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pose_keypoint": ("POSE_KEYPOINT",),
                "LeftKnee_Vertical": ("INT", {"min": -100, "max": 100}),
                "LeftKnee_Horizontal": ("INT", {"min": -100, "max": 100}),
                "RightKnee_Vertical": ("INT", {"min": -100, "max": 100}),
                "RightKnee_Horizontal": ("INT", {"min": -100, "max": 100}),
                "Lefthand_Vertical": ("INT", {"min": -100, "max": 100}),
                "Lefthand_Horizontal": ("INT", {"min": -100, "max": 100}),
                "Righthand_Vertical": ("INT", {"min": -100, "max": 100}),
                "Righthand_Horizontal": ("INT", {"min": -100, "max": 100}),
                "Leftleg_Vertical": ("INT", {"min": -100, "max": 100}),
                "Leftleg_Horizontal": ("INT", {"min": -100, "max": 100}),
                "Rightleg_Vertical": ("INT", {"min": -100, "max": 100}),
                "Rightleg_Horizontal": ("INT", {"min": -100, "max": 100}),
                "image_width": ("INT", {"min": 1, "max": MAX_RESOLUTION}),
                "image_height": ("INT", {"min": 1, "max": MAX_RESOLUTION}),
            },
            "optional": {
                "hand_openpose": ("INT", {"default": 0}),
                "face_openpose": ("INT", {"default": 0}),
                "person_number": ("INT", {"default": 0}),
            }
        }


        
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    RETURN_NAMES = ("image", "POSE_KEYPOINT")
    FUNCTION = "box_keypoints"
    CATEGORY = "openpose"

    def get_keypoint_from_list(self, keypoint_list, item):
        idx_x = item * 3
        idx_y = idx_x + 1
        idx_conf = idx_y + 1
        return keypoint_list[idx_x], keypoint_list[idx_y], keypoint_list[idx_conf]

    def check_keypoints_within_bounds(self, keypoints):
        for i in range(0, len(keypoints), 3):
            x, y = keypoints[i], keypoints[i + 1]
            if not (0 < x < 1 and 0 < y < 1):
                return False
        return True


    def adjust_single(self, number, pose_keypoints, vertical, horizontal, image_width, image_height):
        node_idx = number
            
        node = self.get_keypoint_from_list(pose_keypoints, node_idx)

        horizontal_adjustment = horizontal / 100
        vertical_adjustment = vertical / 100

        if node[2] > 0:
            new_node = (
                node[0] + horizontal_adjustment,
                node[1] + vertical_adjustment,
                node[2]
            )

            node = new_node

            pose_keypoints[node_idx * 3] = node[0]
            pose_keypoints[node_idx * 3 + 1] = node[1]
            pose_keypoints[node_idx * 3 + 2] = node[2]
                    
        return pose_keypoints
        

    def adjust_left_leg(self, Leftorright, pose_keypoints, vertical, horizontal, image_width, image_height):
        # 左手相关关键点索引
        if Leftorright == 'left':
            left_shoulder_idx = 11
            left_elbow_idx = 12
            left_wrist_idx = 13
        elif Leftorright == 'right': 
            left_shoulder_idx = 8
            left_elbow_idx = 9
            left_wrist_idx = 10
            
        # 获取左肩、左肘和左手腕的坐标
        left_shoulder = self.get_keypoint_from_list(pose_keypoints, left_shoulder_idx)
        left_elbow = self.get_keypoint_from_list(pose_keypoints, left_elbow_idx)
        left_wrist = self.get_keypoint_from_list(pose_keypoints, left_wrist_idx)

        # 将输入的像素值转换为相对于图像尺寸的小数
        horizontal_adjustment = horizontal / 100
        vertical_adjustment = vertical / 100

        if left_wrist[2] > 0:
            new_left_wrist = (
                left_wrist[0] + horizontal_adjustment,
                left_wrist[1] + vertical_adjustment,
                left_wrist[2]
            )

            move_ratio = 0.4

            if left_elbow[2] > 0:
                new_left_elbow = (
                    left_elbow[0] + move_ratio * horizontal_adjustment,
                    left_elbow[1] + move_ratio * vertical_adjustment,
                    left_elbow[2]
                )

            left_wrist = new_left_wrist
            left_elbow = new_left_elbow

            pose_keypoints[left_wrist_idx * 3] = left_wrist[0]
            pose_keypoints[left_wrist_idx * 3 + 1] = left_wrist[1]
            pose_keypoints[left_wrist_idx * 3 + 2] = left_wrist[2]

            pose_keypoints[left_elbow_idx * 3] = left_elbow[0]
            pose_keypoints[left_elbow_idx * 3 + 1] = left_elbow[1]
            pose_keypoints[left_elbow_idx * 3 + 2] = left_elbow[2]

                    
        return pose_keypoints
        
        
    def adjust_left_hand(self, Leftorright, pose_keypoints, hand_keypoints, vertical, horizontal, image_width, image_height):
        # 左手相关关键点索引
        if Leftorright == 'left':
            left_shoulder_idx = 5
            left_elbow_idx = 6
            left_wrist_idx = 7
        elif Leftorright == 'right': 
            left_shoulder_idx = 2
            left_elbow_idx = 3
            left_wrist_idx = 4
            
        # 获取左肩、左肘和左手腕的坐标
        left_shoulder = self.get_keypoint_from_list(pose_keypoints, left_shoulder_idx)
        left_elbow = self.get_keypoint_from_list(pose_keypoints, left_elbow_idx)
        left_wrist = self.get_keypoint_from_list(pose_keypoints, left_wrist_idx)

        # 将输入的像素值转换为相对于图像尺寸的小数
        horizontal_adjustment = horizontal / 100
        vertical_adjustment = vertical / 100

        if left_wrist[2] > 0:
            new_left_wrist = (
                left_wrist[0] + horizontal_adjustment,
                left_wrist[1] + vertical_adjustment,
                left_wrist[2]
            )

            move_ratio = 0.5

            if left_elbow[2] > 0:
                new_left_elbow = (
                    left_elbow[0] + move_ratio * horizontal_adjustment,
                    left_elbow[1] + move_ratio * vertical_adjustment,
                    left_elbow[2]
                )

            new_hand_keypoints = hand_keypoints.copy()
            if hand_keypoints.size > 0:
                for i in range(0, len(hand_keypoints), 3):
                    new_hand_keypoints[i] = hand_keypoints[i] + horizontal_adjustment
                    new_hand_keypoints[i + 1] = hand_keypoints[i + 1] + vertical_adjustment
                
            # 添加左手腕和左肘关键点到新手部关键点列表中用于检查（不包括置信度值）
            #combined_keypoints = np.append(new_hand_keypoints, [new_left_wrist[0], new_left_wrist[1], 0.5,new_left_elbow[0], new_left_elbow[1],0.5)

            # 检查所有关键点是否在0到1之间
            #if self.check_keypoints_within_bounds(combined_keypoints):
            hand_keypoints = new_hand_keypoints
            left_wrist = new_left_wrist
            left_elbow = new_left_elbow

            pose_keypoints[left_wrist_idx * 3] = left_wrist[0]
            pose_keypoints[left_wrist_idx * 3 + 1] = left_wrist[1]
            pose_keypoints[left_wrist_idx * 3 + 2] = left_wrist[2]

            pose_keypoints[left_elbow_idx * 3] = left_elbow[0]
            pose_keypoints[left_elbow_idx * 3 + 1] = left_elbow[1]
            pose_keypoints[left_elbow_idx * 3 + 2] = left_elbow[2]

                    
        return pose_keypoints, hand_keypoints


    def box_keypoints(self, pose_keypoint, LeftKnee_Vertical,LeftKnee_Horizontal,RightKnee_Vertical,RightKnee_Horizontal, Lefthand_Vertical, Lefthand_Horizontal,Righthand_Vertical,Righthand_Horizontal, Leftleg_Vertical, Leftleg_Horizontal,Rightleg_Vertical, Rightleg_Horizontal, image_width, image_height, person_number=0, hand_openpose=0, face_openpose=0):
    
            # 准备调试信息
        debug_info = []


        # 重新从输入数据中加载原始pose_keypoint
        original_pose_keypoint = copy.deepcopy(pose_keypoint)
        
        
        people = original_pose_keypoint[0].get("people", [])
        if person_number >= len(people):
            return self.solid(0, image_width, image_height)[0], pose_keypoint  # Handle case where person_number is out of range

        # 获取第一个人的关键点
        pose_keypoints = np.array(people[person_number]["pose_keypoints_2d"])
        pose_keypoints_all = np.array(people[person_number])

        # 如果 hand_openpose 为 True，获取手部关键点
        left_hand_keypoints = np.array([])
        right_hand_keypoints = np.array([])
        if hand_openpose > 0:
            debug_info.append("hand_openpose_succ")
            left_hand_keypoints = np.array(people[person_number]["hand_left_keypoints_2d"])
            right_hand_keypoints = np.array(people[person_number]["hand_right_keypoints_2d"])


        debug_info.append("all:\n" + str(pose_keypoints_all))
        debug_info.append("pose_keypoints:\n" + str(pose_keypoints))
        debug_info.append("left_hand_keypoints:\n" + str(left_hand_keypoints))
        debug_info.append("right_hand_keypoints:\n" + str(right_hand_keypoints))

        # 如果 face_openpose 为 True，获取面部关键点
        face_keypoints = np.array([])
        if face_openpose > 0:
            debug_info.append("face_openpose_succ")
            face_keypoints = np.array(people[person_number].get("face_keypoints_2d", []))
            debug_info.append("face_keypoints:\n" + str(face_keypoints))

      
        # 调整关键点 
        pose_keypoints = self.adjust_single(12,pose_keypoints, LeftKnee_Vertical, LeftKnee_Horizontal, image_width, image_height)
        pose_keypoints = self.adjust_single(9,pose_keypoints, RightKnee_Vertical, RightKnee_Horizontal, image_width, image_height)
        
        pose_keypoints, left_hand_keypoints = self.adjust_left_hand('left',pose_keypoints, left_hand_keypoints, Lefthand_Vertical, Lefthand_Horizontal, image_width, image_height)

        pose_keypoints, right_hand_keypoints = self.adjust_left_hand('right',pose_keypoints, right_hand_keypoints, Righthand_Vertical, Righthand_Horizontal, image_width, image_height)

        pose_keypoints = self.adjust_left_leg('left',pose_keypoints, Leftleg_Vertical, Leftleg_Horizontal, image_width, image_height)
        pose_keypoints = self.adjust_left_leg('right',pose_keypoints, Rightleg_Vertical, Rightleg_Horizontal, image_width, image_height)    
        
        
        # 更新原始关键点
        people[person_number]["pose_keypoints_2d"] = pose_keypoints.tolist()
        if hand_openpose > 0  and left_hand_keypoints.size > 0:
            people[person_number]["hand_left_keypoints_2d"] = left_hand_keypoints.tolist()
        if hand_openpose > 0  and right_hand_keypoints.size > 0:
            people[person_number]["hand_right_keypoints_2d"] = right_hand_keypoints.tolist()
        if face_openpose> 0 and face_keypoints.size > 0:
            people[person_number]["face_keypoints_2d"] = face_keypoints.tolist()

        # 保留原有的 canvas_width 和 canvas_height
        original_canvas_width = pose_keypoint[0].get("canvas_width", image_width)
        original_canvas_height = pose_keypoint[0].get("canvas_height", image_height)
        
        # 重新组装新的 pose_keypoint 数据结构
        new_pose_keypoint = [{"people": people, "canvas_width": original_canvas_width, "canvas_height": original_canvas_height}]
        
        # 创建绘制所需的 pose 数据
        bodies = dict(candidate=pose_keypoints.reshape(-1, 3).tolist(), subset=[np.arange(len(pose_keypoints) // 3).tolist()])
        hands = []
        if hand_openpose>0:
            #if left_hand_keypoints.ndim == 2 and left_hand_keypoints.shape[1] >= 2:
            processed_left_hand_keypoints = np.array(left_hand_keypoints).reshape(-1, 3).tolist()
            processed_left_hand_keypoints = [[x, y] for x, y, _ in processed_left_hand_keypoints]
            hands.append(processed_left_hand_keypoints)
            processed_right_hand_keypoints = np.array(right_hand_keypoints).reshape(-1, 3).tolist()
            processed_right_hand_keypoints = [[x, y] for x, y, _ in processed_right_hand_keypoints]
            hands.append(processed_right_hand_keypoints)

        faces = []
        if face_openpose > 0 and face_keypoints.size > 0:
            processed_face = np.array(face_keypoints).reshape(-1, 3).tolist()
            processed_face = [[x, y] for x, y, _ in processed_face]
            faces.append(processed_face)
            
        pose = dict(bodies=bodies, hands=hands, faces=faces)


        
        if hand_openpose > 0:
            hand_openpose_t = True
            #debug_info.append("hand_openpose_t_true:\n" )
        else:
            hand_openpose_t = False
        if face_openpose > 0:
            face_openpose_t = True
        else:
            face_openpose_t = False
            
        #tmp img
        img = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
        img = resize_image(img, 512)
        H, W, C = img.shape
        
        canvas = draw_pose(pose, H, W, draw_body=True, draw_hand=hand_openpose_t, draw_face=face_openpose_t)
        output_pil = Image.fromarray(canvas)
        output_pil.save('testoutput.png')  # 保存图像
   
        debug_info.append("HHH:\n" + str(H))
        debug_info.append("WWW:\n" + str(W))
        
        numpy_image = np.array(output_pil).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(numpy_image)[None, :]  # 添加批量维度

        #detected_map = HWC3(canvas)
        #detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        #detected_map = Image.fromarray(detected_map)
        

        #canvas_float = detected_map.astype(np.float32) / 255.0
        
        #debug_info.append(f"canvas_float:\n{canvas_float.shape}")
                
                
        #canvas_float = np.stack(canvas_float, axis=0)
        #tensor = torch.tensor(canvas_float)
        


        # 将调试信息写入文件
        with open("debug_info.txt", "w") as f:
            f.write("\n\n".join(debug_info))
            
        return tensor_image , new_pose_keypoint
        #return (tensor,), new_pose_keypoint



    def solid(self, value, width, height):
        out = torch.full((1, 3, height, width), value, dtype=torch.float32, device="cpu")
        return (out,)





@torch.no_grad()
def match_normalize(target_tensor, source_tensor, strength=1.0, dimensions=4):
    "Adjust target_tensor based on source_tensor's mean and stddev"   
    if len(target_tensor.shape) != dimensions:
        raise ValueError("source_latent must have four dimensions")
    if len(source_tensor.shape) != dimensions:
        raise ValueError("target_latent must have four dimensions")

    # Put everything on the same device
    device = target_tensor.device

    # Calculate the mean and std of target tensor
    tgt_mean = target_tensor.mean(dim=[2, 3], keepdim=True).to(device)
    tgt_std = target_tensor.std(dim=[2, 3], keepdim=True).to(device)
    
    # Calculate the mean and std of source tensor
    src_mean = source_tensor.mean(dim=[2, 3], keepdim=True).to(device)
    src_std = source_tensor.std(dim=[2, 3], keepdim=True).to(device)
    
    # Normalize target tensor to have mean=0 and std=1, then rescale with strength adjustment
    normalized_tensor = (target_tensor.clone() - tgt_mean) / tgt_std * src_std + src_mean
    adjusted_tensor = target_tensor + strength * (normalized_tensor - target_tensor)
    
    return adjusted_tensor


@register_node("LatentMatch", "KW Latent Match")
class LatentMatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"source_latent": ("LATENT", ),
                     "target_latent": ("LATENT", ),
                     "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1})}}
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "tensor_ops"

    FUNCTION = "latent_match"

    @torch.no_grad()
    def latent_match(self, source_latent, target_latent, strength=1.0):       
        normalized_latent = match_normalize(target_latent["samples"], source_latent["samples"], strength, dimensions=4)

        return_latent = source_latent.copy()
        return_latent["samples"] = normalized_latent
        return (return_latent,)