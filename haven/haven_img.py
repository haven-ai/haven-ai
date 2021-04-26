import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pylab as plt
from skimage.segmentation import mark_boundaries

from . import haven_utils as hu


def mask_on_image(image, mask, add_bbox=False, return_pil=False):
    """[summary]

    Parameters
    ----------
    image : [type]
        [description]
    mask : [type]
        [description]
    add_bbox : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    image = image_as_uint8(image)
    mask = np.array(mask).squeeze()
    obj_ids = np.unique(mask)

    # polygons = cv2.findContours(im,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)[1][0]
    red = np.zeros(image.shape, dtype="uint8")
    red[:, :, 2] = 255
    alpha = 0.5
    result = image.copy()
    for o in obj_ids:
        if o == 0:
            continue
        ind = mask == o
        result[ind] = result[ind] * alpha + red[ind] * (1 - alpha)
        pos = np.where(ind)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        if add_bbox:
            result = cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
    result = mark_boundaries(result, mask)

    if return_pil:
        return Image.fromarray(result)

    return result


def resize_points(points, h, w):
    """[summary]

    Parameters
    ----------
    points : [type]
        [description]
    h : [type]
        [description]
    w : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    points = points.squeeze()
    h_old, w_old = points.shape
    y_list, x_list = np.where(points.squeeze())

    points_new = np.zeros((h, w))

    for y, x in zip(y_list, x_list):
        y_new = int((y / h_old) * h)
        x_new = int((x / w_old) * w)
        points_new[y_new, x_new] = 1

    return points_new


def gray2cmap(gray, cmap="jet", thresh=0):
    """gets a heatmap for a given gray image. Can be used to visualize probabilities.

    Parameters
    ----------
    gray : [type]
        [description]
    cmap : str, optional
        [description], by default "jet"
    thresh : int, optional
        [description], by default 0

    Returns
    -------
    [type]
        [description]
    """
    # Gray has values between 0 and 255 or 0 and 1
    gray = hu.t2n(gray)
    gray = gray / max(1, gray.max())
    gray = np.maximum(gray - thresh, 0)
    gray = gray / max(1, gray.max())
    gray = gray * 255

    gray = gray.astype(int)
    # print(gray)

    from matplotlib.cm import get_cmap

    cmap = get_cmap(cmap)

    output = np.zeros(gray.shape + (3,), dtype=np.float64)

    for c in np.unique(gray):
        output[(gray == c).nonzero()] = cmap(c)[:3]

    return hu.l2f(output)


def scatter_plot(X, color, fig=None, title=""):
    """[summary]

    Parameters
    ----------
    X : [type]
        [description]
    color : [type]
        [description]
    fig : [type], optional
        [description], by default None
    title : str, optional
        [description], by default ""

    Returns
    -------
    [type]
        [description]
    """
    if fig is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)

    ax = fig.axes[0]

    ax.grid(linestyle="dotted")
    ax.scatter(X[:, 0], X[:, 1], alpha=0.6, c=color, edgecolors="black")

    # plt.axes().set_aspect('equal', 'datalim')
    ax.set_title(title)
    ax.set_xlabel("t-SNE Feature 2")
    ax.set_ylabel("t-SNE Feature 1")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig


def pretty_vis(image, annList, show_class=False, alpha=0.0, dpi=100, **options):
    import cv2
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.patches import Polygon
    from matplotlib.figure import Figure
    from . import ann_utils as au

    # print(image)
    # if not image.as > 1:
    #     image = image.astype(float)/255.
    image = f2l(image).squeeze().clip(0, 255)
    if image.max() > 1:
        image /= 255.0

    # box_alpha = 0.5
    # print(image.clip(0, 255).max())
    color_list = colormap(rgb=True) / 255.0

    # fig = Figure()
    fig = plt.figure(frameon=False)
    canvas = FigureCanvas(fig)
    fig.set_size_inches(image.shape[1] / dpi, image.shape[0] / dpi)
    # ax = fig.gca()

    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    fig.add_axes(ax)
    # im = im.clip(0, 1)
    # print(image)
    ax.imshow(image)

    # Display in largest to smallest order to reduce occlusion
    # areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in range(len(annList)):
        ann = annList[i]

        # bbox = boxes[i, :4]
        # score = boxes[i, -1]

        # bbox = au.ann2bbox(ann)["shape"]
        # score = ann["score"]

        # if score < thresh:
        #     continue

        # show box (off by default, box_alpha=0.0)
        if "bbox" in ann:
            bbox = ann["bbox"]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor="r", linewidth=3.0, alpha=0.5)
            )

        # if show_class:
        # if options.get("show_text") == True or options.get("show_text") is None:
        #     score = ann["score"] or -1
        #     ax.text(
        #         bbox[0], bbox[1] - 2,
        #         "%.1f" % score,
        #         fontsize=14,
        #         family='serif',
        #         bbox=dict(facecolor='g', alpha=1.0, pad=0, edgecolor='none'),
        #         color='white')

        # show mask
        if "segmentation" in ann:
            mask = au.ann2mask(ann)["mask"]
            img = np.ones(image.shape)
            # category_id = ann["category_id"]
            # mask_color_id = category_id - 1
            # color_list = ["r", "g", "b","y", "w","orange","purple"]
            # color_mask = color_list[mask_color_id % len(color_list)]
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            # print("color id: %d - category_id: %d - color mask: %s"
            # %(mask_color_id, category_id, str(color_mask)))
            w_ratio = 0.4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = mask

            contour, hier = cv2.findContours(e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)), fill=True, facecolor=color_mask, edgecolor="white", linewidth=1.5, alpha=0.7
                )
                ax.add_patch(polygon)

    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

    fig_image = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(int(height), int(width), 3)
    plt.close()
    # print(fig_image)
    return fig_image


def text_on_image(text, image):
    """Adds test on the image

    Parameters
    ----------
    text : [type]
        [description]
    image : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 40)
    fontScale = 0.8
    fontColor = (1, 1, 1)
    lineType = 1
    # img_mask = skimage.transform.rescale(np.array(img_mask), 1.0)
    # img_np = skimage.transform.rescale(np.array(img_points), 1.0)
    img_np = cv2.putText(
        image,
        text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness=2
        # lineType
    )
    return img_np


def bbox_on_image(bbox_xyxy, image, mode="yxyx", color=(255, 0, 0)):
    """[summary]

    Parameters
    ----------
    bbox_xyxy : [type]
        [description]
    image : [type]
        [description]
    mode : str, optional
        [description], by default 'xyxy'

    Returns
    -------
    [type]
        [description]
    """
    image_uint8 = image_as_uint8(image)

    H, W, _ = image_uint8.shape

    for bb in bbox_xyxy:
        if mode == "yxyx":
            y1, x1, y2, x2 = bb
        else:
            x1, y1, x2, y2 = bb

        if mode == "xywh":
            x2 += x1
            y2 += y1

        if x2 < 1:
            start_point = (
                int(x1 * W),
                int(y1 * H),
            )
            end_point = (
                int(x2 * W),
                int(y2 * H),
            )
        else:
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))

        # Line thickness of 2 px
        thickness = 2
        # Draw a rectangle with blue line borders of thickness of 2 px
        image_uint8 = cv2.rectangle(image_uint8.copy(), start_point, end_point, color, thickness)

    return image_uint8 / 255.0


def points_on_image(y_list, x_list, image, radius=3, c_list=None):
    """[summary]

    Parameters
    ----------
    y_list : [type]
        [description]
    x_list : [type]
        [description]
    image : [type]
        [description]
    radius : int, optional
        [description], by default 3

    Returns
    -------
    [type]
        [description]
    """
    image_uint8 = image_as_uint8(image)

    H, W, _ = image_uint8.shape
    color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, (y, x) in enumerate(zip(y_list, x_list)):
        if y < 1:
            x, y = int(x * W), int(y * H)
        else:
            x, y = int(x), int(y)

        # Blue color in BGR
        if c_list is not None:
            color = color_list[c_list[i]]
        else:
            color = color_list[1]

        # Line thickness of 2 px
        thickness = 5
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        image_uint8 = cv2.circle(image_uint8, (x, y), radius, color, thickness)

        start_point = (x - radius * 2, y - radius * 2)
        end_point = (x + radius * 2, y + radius * 2)
        thickness = 2
        color = (255, 0, 0)

        image_uint8 = cv2.rectangle(image_uint8, start_point, end_point, color, thickness)

    return image_uint8 / 255.0


def image_as_uint8(img):
    """Returns a uint8 version of the image

    Parameters
    ----------
    img : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    image = hu.f2l(np.array(img).squeeze())

    if image.dtype != "uint8":
        image_uint8 = (image * 255).astype("uint8").copy()
    else:
        image_uint8 = image

    return image_uint8
