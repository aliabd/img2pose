
import gdown
import gradio
import zipfile
import subprocess
import numpy as np
from torchvision import transforms
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os

os.chdir("Sim3DR")
subprocess.call(['sh', './build_sim3dr.sh'])
os.chdir("../")

from utils.renderer import Renderer
from img2pose import img2poseModel
from model_loader import load_model


if not os.path.exists("models/models.zip"):
    os.mkdir("models")
    gdown.download("https://drive.google.com/uc?id=1OvnZ7OUQFg2bAgFADhT7UnCkSaXst10O", "models/models.zip")
    with zipfile.ZipFile("models/models.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

renderer = Renderer(
    vertices_path="pose_references/vertices_trans.npy",
    triangles_path="pose_references/triangles.npy"
)

threed_points = np.load('pose_references/reference_3d_68_points_trans.npy')


def render_plot(img, poses, bboxes, with_bounding):
    (w, h) = img.size
    image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])

    trans_vertices = renderer.transform_vertices(img, poses)
    img = renderer.render(img, trans_vertices, alpha=1)

    plt.figure()

    if with_bounding:
        for bbox in bboxes:
            plt.gca().add_patch(
                patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=3, edgecolor='b',
                                  facecolor='none'))
    plt.axis('off')
    plt.imshow(img)
    return plt


transform = transforms.Compose([transforms.ToTensor()])

DEPTH = 18
MAX_SIZE = 1400
MIN_SIZE = 600

POSE_MEAN = "models/WIDER_train_pose_mean_v1.npy"
POSE_STDDEV = "models/WIDER_train_pose_stddev_v1.npy"
MODEL_PATH = "models/img2pose_v1.pth"

pose_mean = np.load(POSE_MEAN)
pose_stddev = np.load(POSE_STDDEV)

img2pose_model = img2poseModel(
    DEPTH, MIN_SIZE, MAX_SIZE,
    pose_mean=pose_mean, pose_stddev=pose_stddev,
    threed_68_points=threed_points,
)
load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
img2pose_model.evaluate()


def generate(img, with_bounding):
    threshold = 0.9
    img = img.convert("RGB")
    (w, h) = img.size
    image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
    res = img2pose_model.predict([transform(img)])[0]
    all_bboxes = res["boxes"].cpu().numpy().astype('float')
    poses = []
    bboxes = []
    for i in range(len(all_bboxes)):
        if res["scores"][i] > threshold:
            bbox = all_bboxes[i]
            pose_pred = res["dofs"].cpu().numpy()[i].astype('float')
            pose_pred = pose_pred.squeeze()

            poses.append(pose_pred)
            bboxes.append(bbox)

    return render_plot(img.copy(), poses, bboxes, with_bounding)

title = "img2pose: Face Alignment and Detection via 6DoF, Face Pose Estimation"
description = "This is a web demo of img2pose. To use it upload your own image or click one of the examples below. " \
              "For more information, see the links at the bottom."
examples = [
    ["example_images/2.jpg", False],
    ["example_images/3.jpg", False],
    ["example_images/1.jpg", True]
]

article = """
<p style='text-align: center'>This demo is based on the <a href="https://arxiv.org/abs/2012.07791">img2pose: Face Alignment and Detection via 6DoF, Face Pose Estimation</a> paper</p>
<p style='text-align: center'>For more info see the <a href="https://github.com/vitoralbiero/img2pose">Github Repo</a> </p>
<p style='text-align: center'>This work is licensed under the <a href="https://github.com/vitoralbiero/img2pose/blob/main/license.md">Attribution-NonCommercial 4.0 International License</a></p>
"""

inputs = [gradio.inputs.Image(type="pil", label="Your Image"), gradio.inputs.Checkbox(label="Include Bounding Boxes?")]
outputs = gradio.outputs.Image(label="Output Image", type="plot")
gradio.Interface(generate, inputs, outputs, title=title, description=description, examples=examples,
                 article=article, allow_flagging=False).launch()
