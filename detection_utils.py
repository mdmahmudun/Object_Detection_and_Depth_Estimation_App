import io
import matplotlib.pyplot as plt
import inflect
from PIL import Image


def render_results_in_image(in_pil_img, in_results):
    plt.figure(figsize=(16,10))
    plt.imshow(in_pil_img)

    ax = plt.gca()

    for prediction in in_results:
        x, y = prediction['box']['xmin'], prediction['box']['ymin']
        w = prediction['box']['xmax'] - prediction['box']['xmin']
        h = prediction['box']['ymax'] - prediction['box']['ymin']

        ax.add_patch(plt.Rectangle(
            (x,y),
            w,
            h,
            fill = False,
            color = 'green',
            linewidth = 2
        ))

        ax.text(
            x,
            y,
            f"{prediction['label']}: {round(prediction['score']*100,1)}%",
            color = 'red'
        )
    plt.axis('off')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format = 'png', bbox_inches = 'tight', pad_inches = 0)
    img_buf.seek(0)
    modified_image = Image.open(img_buf)
    plt.close()
    return modified_image


def summarize_predictions_natural_language(predictions):
    summary = {}
    p = inflect.engine()

    for prediction in predictions:
        label = prediction['label']
        if label in summary:
            summary[label] += 1
        else:
            summary[label] = 1

    result_string = "In this image, there are "
    for i, (label, count) in enumerate(summary.items()):
        count_string = p.number_to_words(count)
        result_string += f"{count_string} {label}"

        if count > 1:
            result_string += 's'
        result_string += " "

        if i == len(summary) - 2:
            result_string += "and "
    result_string = result_string.rstrip(', ') + "."
    return result_string