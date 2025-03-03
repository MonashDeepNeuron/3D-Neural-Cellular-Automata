#@title Visualization code
#@title Required imports
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import seaborn as sns

from IPython.display import display, HTML
from moviepy.editor import *

import json

#@title Utils functions

def dec_to_rgb(dec_color):
  return (dec_color // 256 // 256 % 256, dec_color // 256 % 256, dec_color % 256)

def rgb_to_hex(r, g, b):
  return ('#{:02X}{:02X}{:02X}').format(int(r), int(g), int(b))

def norm_rgb(input):
  input[:, :, :, :3] = input[:, :, :, :3] / 255.
  return input

def denorm_rgb(input):
  input[:, :, :, :3] = (input[:, :, :, :3] * 255).astype(np.int32)
  return input

def convert_size(size_bytes):
  import math
  if size_bytes == 0:
      return "0B"
  size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
  i = int(math.floor(math.log(size_bytes, 1024)))
  p = math.pow(1024, i)
  s = round(size_bytes / p, 2)
  return "%s %s" % (s, size_name[i])


# --- Plotly visualization ---

def create_voxel_mesh(x, y, z, color, opacity=True):
  r, g, b, a = np.clip(color, 0, 255)
  a = np.round(np.clip(a, 0, 1), 2) if opacity else 1
  
  rgba_str = "rgba({}, {}, {}, {})".format(r, g, b, a)

  mesh = go.Mesh3d(
      x=np.array([0, 0, 1, 1, 0, 0, 1, 1]) + x,
      y=np.array([0, 1, 1, 0, 0, 1, 1, 0]) + y,
      z=np.array([0, 0, 0, 0, 1, 1, 1, 1]) + z,
      i=[7, 0, 0, 0, 4, 4, 6, 1, 4, 0, 3, 6],
      j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
      k=[0, 7, 2, 3, 6, 7, 1, 6, 5, 5, 7, 2],
      color = rgba_str,
      flatshading=True
  )

  return mesh

def plot_voxel_grid(grid, space_shape=None, th=None, opacity=True, image=False):
  if space_shape is None:
    space_shape = grid.shape[:3]

  data = plot_single_instance(grid, th, opacity)

  fig = go.Figure(data=data)
  fig.update_layout(scene_aspectmode="cube")
  fig.update_scenes(camera_up_z=0, camera_up_y=1)
  fig.update_layout(
    scene = dict(
        xaxis = dict(range=[0, space_shape[2]]),
        yaxis = dict(range=[0, space_shape[1]]),
        zaxis = dict(range=[0, space_shape[0]]),
    )
  )
  if image:
    return fig.to_image(format="png")
  fig.show()

def plot_single_instance(instance, th=None, opacity=True):
  data = []

  for z in range(instance.shape[0]):
    for y in range(instance.shape[1]):
      for x in range(instance.shape[2]):
        if len(instance.shape) < 4:
          if instance[z, y, x] == 1:
            mesh = create_voxel_mesh(x, y, z, [127, 127, 127, 1])
            data.append(mesh)
        else:
          # If the alpha value is lower than the threshold don't show it
          if th and instance[z, y, x][3] < th:
            continue
          
          color = instance[z, y, x][:4]

          mesh = create_voxel_mesh(x, y, z, color, opacity)
          data.append(mesh)
  
  return data

def animate_run(run, th=None, opacity=True):
  frames = []
  for i, x in enumerate(run[1:]):
    title = "Iteration {}".format(i)
    data = plot_single_instance(x, th, opacity)
    frame = go.Frame(data=data, layout=go.Layout(title_text=title))
    frames.append(frame)

  fig = go.Figure(
      data=plot_single_instance(run[0], th, opacity),
      layout=go.Layout(
          xaxis=dict(range=[0, 5], autorange=False),
          yaxis=dict(range=[0, 5], autorange=False),
          title="Start Title",
          updatemenus=[dict(
              type="buttons",
              buttons=[dict(label="Play",
                            method="animate",
                            args=[None])])]
      ),
      frames=frames
  )
  fig.update_layout(scene_aspectmode="cube")
  fig.update_scenes(camera_up_z=0, camera_up_y=1)
  shape = run[0].shape
  fig.update_layout(
    scene = dict(
        xaxis = dict(range=[0, shape[2]]),
        yaxis = dict(range=[0, shape[1]]),
        zaxis = dict(range=[0, shape[0]]),
    )
  )

  fig.show()

# --- End Plotly visualization ---

def show_three_visualizer(array):
  """
  Use the three.js library in order to show the provided `array`.
  `array` can be a 3D matrix or a 4D matrix.
  In this last case, it will be interpreted as an animation of different 3D frames.
  """
  array = array[..., :4]
  array = np.clip(array, 0, 255)
  python_data = array.tolist()
  html_data = """
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.101.1/examples/js/controls/OrbitControls.js"></script>

  <script>
    class VoxelVisualizer {
      constructor(width, height, data, frameDuration) {
        this.width = width;
        this.height = height;
        this.data = data;
        this.frameDuration = frameDuration;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xffffff);

        this.camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);

        this.renderer = new THREE.WebGLRenderer({preserveDrawingBuffer: true});
        this.renderer.setSize(width, height);
        document.body.appendChild(this.renderer.domElement);

        this.scene.add(new THREE.AxesHelper(width));

        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);

        this.cubeGeometry = new THREE.BoxGeometry();
        this.voxels = [];

        let firstFrame;
        if (Array.isArray(this.data[0][0][0][0])) {
          // Animation
          firstFrame = this.data[0];

          this.slider = this.createSlider();
          this.playToggle = this.createPlayToggle();
          this.screenBtn = this.createScreenshotButton();
          this.counter = this.createCounter();

          document.body.appendChild(this.playToggle);
          document.body.appendChild(this.screenBtn);
          document.body.appendChild(this.slider);
          document.body.appendChild(this.counter);

          this.playing = false;
          this.framesInterval = null;
          
        } else {
          // Single voxel grid
          firstFrame = this.data;
        }

        this.drawVoxels(firstFrame);

        const cameraDistance = 5;
        this.camera.position.set(
          firstFrame[0][0].length + cameraDistance, 
          firstFrame[0].length + cameraDistance, 
          firstFrame.length + cameraDistance
        );
        this.controls.update();

        this.animate();
      }

      drawVoxel(x, y, z, color, opacity) {
        if (opacity < 0.1) return;
        const material = new THREE.MeshBasicMaterial({ color: color, opacity: opacity, transparent: true });
        const voxel = new THREE.Mesh(this.cubeGeometry, material);
        voxel.position.set(x, y, z);
        this.scene.add(voxel);
        this.voxels.push(voxel);
      }

      clearScene() {
        for (const voxel of this.voxels) {
          voxel.material.dispose();
          this.scene.remove(voxel);
        }
        this.voxels = [];
      }

      drawVoxels(grid) {
        for (let z = 0; z < grid.length; z++) {
          for (let y = 0; y < grid[0].length; y++) {
            for (let x = 0; x < grid[0][0].length; x++) {
              const [r, g, b, a] = grid[z][y][x];
              this.drawVoxel(x, y, z, `rgba(${r}, ${g}, ${b})`, a);
            }
          }
        }
      }

      createPlayToggle() {
        const playToggle = document.createElement("button");
        playToggle.textContent = "Play";
        playToggle.style.width = "50px";
        playToggle.onclick = () => {
          this.playing = !this.playing;
          if (this.playing) {
            this.startAnimation(this.slider.value);
            playToggle.textContent = "Pause";
          }
          else {
            this.stopAnimation();
          }
        }
        return playToggle;
      }

      createScreenshotButton() {
        const btn = document.createElement("button");
        btn.textContent = "Screen";
        btn.style.width = "50px";
        btn.onclick = () => this.saveAsImage();
        return btn;
      }

      createSlider() {
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = 0;
        slider.max = this.data.length - 1;
        slider.oninput = () => {
          const currentFrame = slider.value;
          this.counter.textContent = currentFrame;
          this.clearScene();
          this.drawVoxels(this.data[currentFrame]);
        }
        slider.value = 0;
        slider.style.width = "445px";
        return slider;
      }

      createCounter() {
        const counter = document.createElement("span");
        counter.textContent = 0;
        return counter;
      }

      startAnimation(startingFrame) {
        // Reset animation if it was finished
        if (startingFrame == this.data.length - 1)
          startingFrame = 0;

        let currentFrame = startingFrame;
        this.framesInterval = setInterval(() => {
          this.clearScene();
          this.slider.value = currentFrame;
          this.counter.textContent = currentFrame;
          this.drawVoxels(this.data[currentFrame++]);
          if (currentFrame == this.data.length) {
            this.stopAnimation();
          }
        }, this.frameDuration);
      }

      stopAnimation() {
        clearInterval(this.framesInterval);
        this.playToggle.textContent = "Play";
        this.playing = false;
      }

      animate = () => {
        requestAnimationFrame(this.animate);
        this.renderer.render(this.scene, this.camera);
      };

      saveAsImage() {
        let imgData, imgNode;

        try {
            let strMime = "image/jpeg";
            imgData = this.renderer.domElement.toDataURL(strMime);
            console.log(imgData);
            let strData = imgData.replace(strMime, "image/octet-stream");
            let filename = "test.jpg";
            let link = document.createElement('a');

            document.body.appendChild(link); //Firefox requires the link to be in the body
            link.download = filename;
            link.href = strData;
            link.click();
            document.body.removeChild(link); //remove the link when done
        } catch (e) {
            console.log(e);
            return;
        }
      }
    }

    new VoxelVisualizer(500, 500, data, 500);
  
  </script>
  """
  final_data = "<script>const data = " + str(python_data) + ";</script>" + html_data
  display(HTML(final_data))


def explode(data):
  """
  Double the size of the space occupied by the data.
  It is used in order to show every voxels in a matplolib 3D plot,
  because otherwise only external voxels are rendered.
  """
  size = np.array(data.shape)*2
  if len(data.shape) > 3:
    size -= 1
    size[-1] = data.shape[-1]
    data_e = np.zeros(size, dtype=data.dtype)
    data_e[::2, ::2, ::2, :] = data
  else:
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
  return data_e

def plot_voxels_matplotlib(grid, ax=None, edgecolor=None, edgecolors=None, axis="off", camera_dist=8, camera_elev=30, camera_azim=-45, render_internal_voxels=False, gaps=0.05, figsize=None):
  if not ax:
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(projection="3d")

  ax.dist = camera_dist
  ax.axis(axis)

  ax.view_init(elev=camera_elev, azim=camera_azim)

  voxels = grid[:, :, :, 3] > 0.1
  facecolors = grid[:, :, :, :4]

  if render_internal_voxels:
    voxels = explode(voxels)
    facecolors = explode(facecolors)
    if edgecolors is not None:
      edgecolors = explode(edgecolors)

    # Shrink the gaps
    x, y, z = np.indices(np.array(voxels.shape) + 1).astype(float) // 2
    x[0::2, :, :] += gaps
    y[:, 0::2, :] += gaps
    z[:, :, 0::2] += gaps
    x[1::2, :, :] += 1 - gaps
    y[:, 1::2, :] += 1 - gaps
    z[:, :, 1::2] += 1 - gaps

    ax.voxels(x, y, z, voxels, facecolors=facecolors, edgecolor=edgecolor, edgecolors=edgecolors, lightsource=LightSource(azdeg=30, altdeg=30))
  else:
    ax.voxels(voxels, facecolors=facecolors, edgecolor=edgecolor, edgecolors=edgecolors, lightsource=LightSource(azdeg=30, altdeg=30))

  # Make voxels cubic, mantaining aspect ratio
  max_lim = max(target.shape)

  current_min, current_max = ax.get_xlim()
  delta = max_lim - (current_max - current_min)
  ax.set_xlim(current_min - (delta / 2), current_max + (delta / 2))

  current_min, current_max = ax.get_ylim()
  delta = max_lim - (current_max - current_min)
  ax.set_ylim(current_min - (delta / 2), current_max + (delta / 2))

  current_min, current_max = ax.get_zlim()
  delta = max_lim - (current_max - current_min)
  ax.set_zlim(current_min - (delta / 2), current_max + (delta / 2))


#@title Load voxel mesh

def load_voxel_mesh(path):
  # with open(path, "r") as f:
  #   mesh_data = json.load(f, encoding='latin-1')
  with open(path, "r", encoding="latin-1") as f:
    mesh_data = json.load(f)
  palette = mesh_data["palettepip install midvoxio"]
  voxels = mesh_data["layers"][0]["voxels"]

  return voxels, palette

def compute_bbox(voxels):
  voxels = np.array(voxels)
  min_x, max_x = np.min(voxels[:, 0]), np.max(voxels[:, 0])
  min_y, max_y = np.min(voxels[:, 1]), np.max(voxels[:, 1])
  min_z, max_z = np.min(voxels[:, 2]), np.max(voxels[:, 2])
  return (min_x, min_y, min_z), (max_x, max_y, max_z)

def make_voxel_grid_center(voxels, palette, padding=0):
  rgb_palette = [dec_to_rgb(dec) for dec in palette]

  voxels = np.array(voxels)
  (min_x, min_y, min_z), (max_x, max_y, max_z) = compute_bbox(voxels)
  depth = max_z - min_z + 1 + 2*padding
  height = max_y - min_y + 1 + 2*padding
  width = max_x - min_x + 1 + 2*padding
  voxel_grid = np.zeros((depth, height, width, 4))

  for voxel in voxels:
    x, y, z, palette_index = voxel
    x -= min_x
    y -= min_y
    z -= min_z
    color = rgb_palette[palette_index]
    voxel_grid[padding+z, padding+y, padding+x][:3] = np.array(color)
    voxel_grid[padding+z, padding+y, padding+x][3] = 1

  return voxel_grid

# voxels, palette = load_voxel_mesh("voxel_models/donut.vox")
# print("Loaded voxel mesh with {} voxels".format(len(voxels)))
# target = make_voxel_grid_center(voxels, palette, padding=2)
# print("Generated voxel grid of shape", target.shape)
# show_three_visualizer(target)

from midvoxio.voxio import vox_to_arr,viz_vox

print(vox_to_arr('voxel_models/donut.vox').shape)
viz_vox('voxel_models/donut.vox')