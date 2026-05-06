from PIL import Image


from PIL import Image

def resize_keep_aspect(image: Image.Image, longest_side=640):
    w, h = image.size

    if max(w, h) <= longest_side:
        return image  # no need to resize

    if w > h:
        new_w = longest_side
        new_h = int(h * longest_side / w)
    else:
        new_h = longest_side
        new_w = int(w * longest_side / h)

    return image.resize((new_w, new_h), Image.BILINEAR)