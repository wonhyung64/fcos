import tensorflow as tf
from PIL import ImageDraw


def draw_output(
    image,
    final_bboxes,
    final_labels,
    final_scores,
    labels,
    colors,
):
    image = tf.keras.preprocessing.image.array_to_img(image)
    draw = ImageDraw.Draw(image)

    for index, bbox in enumerate(final_bboxes):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)
        label_index = int(final_labels[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels.names[label_index], final_scores[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    return image
