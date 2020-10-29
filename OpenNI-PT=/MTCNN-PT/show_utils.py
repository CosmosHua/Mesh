from PIL import ImageDraw, Image


def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """Draw bounding boxes and facial landmarks.

    Arguments:
        img: image array
        bounding_boxes: a float numpy array of shape [n, 5].
        facial_landmarks: a float numpy array of shape [n, 10].

    Returns:
        an instance of PIL.Image.
    """

    pil_im = Image.fromarray(img[:,:,::-1])
    draw = ImageDraw.Draw(pil_im)

    for b in bounding_boxes:
        draw.rectangle([(b[0],b[1]), (b[2],b[3])], outline='white')

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([
                (p[i] - 1.0, p[i + 5] - 1.0),
                (p[i] + 1.0, p[i + 5] + 1.0)], outline='blue')
    return pil_im
