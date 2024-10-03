@register_node("KwtoolsetImageSelect", "KW Image Select")
class ImageSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "default_img": ("IMAGE",),  # Default image input (required)
            },
            "optional": {
                "img1": ("IMAGE",),  # Optional Image 1
                "img2": ("IMAGE",),  # Optional Image 2
                "img3": ("IMAGE",),  # Optional Image 3
                "img4": ("IMAGE",),  # Optional Image 4
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("selected_img",)
    FUNCTION = "select_image"
    CATEGORY = "advanced/image_processing"

    def select_image(self, default_img=None, img1=None, img2=None, img3=None, img4=None):
        # Check which image to return (img1 > img2 > img3 > img4 > default_img)
        if img1 is not None:
            return (img1,)
        elif img2 is not None:
            return (img2,)
        elif img3 is not None:
            return (img3,)
        elif img4 is not None:
            return (img4,)
        elif default_img is not None:
            return (default_img,)
        else:
            raise ValueError("No valid image provided. All image inputs are None.")
