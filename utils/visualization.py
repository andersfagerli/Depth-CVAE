import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from typing import List


def plot_comparison(comparisons: List[List[np.array]]):
    cols = len(comparisons[0])
    assert cols == 4 or cols == 3,\
         f'Each comparison must contain 4 images (rgb, depth_gt, depth, b) or 3 images (rgb, depth, b)'

    plot_rgb_depth_prediction_uncertainty(comparisons)


def plot_rgb_depth_prediction_uncertainty(comparisons):
    cols = len(comparisons[0])
    rows = len(comparisons)

    fig, ax = plt.subplots(rows, cols)

    for i in range(rows):
        for j in range(cols):
            if rows == 1:
                ax[j].xaxis.set_visible(False)
                ax[j].yaxis.set_visible(False)
                ax[j].set_aspect('equal')
                
                if j == 0:
                    ax[j].imshow(comparisons[i][j])
                    ax[j].text(0.5,-0.2, "RGB", size=12, ha="center", transform=ax[j].transAxes)
                elif j == 1:
                    ax[j].imshow(comparisons[i][j], cmap='hot')
                    ax[j].text(0.5,-0.2, "Ground truth", size=12, ha="center", transform=ax[j].transAxes)
                elif j == 2:
                    ax[j].imshow(comparisons[i][j], cmap='hot')
                    ax[j].text(0.5,-0.2, "Prediction", size=12, ha="center", transform=ax[j].transAxes)
                elif j == 3:
                    ax[j].imshow(comparisons[i][j], cmap='inferno')
                    ax[j].text(0.5,-0.2, "Uncertainty", size=12, ha="center", transform=ax[j].transAxes)   
            else:
                if j == 0:
                    ax[i][j].imshow(comparisons[i][j])
                elif j == 3:
                    ax[i][j].imshow(comparisons[i][j], cmap='inferno')
                else:
                    ax[i][j].imshow(comparisons[i][j], cmap='hot')

                ax[i][j].xaxis.set_visible(False)
                ax[i][j].yaxis.set_visible(False)
                ax[i][j].set_aspect('equal')

                if i == rows - 1:
                    if j == 0:
                        ax[i][j].text(0.5,-0.2, "RGB", size=12, ha="center", transform=ax[i][j].transAxes)
                    elif j == 1:
                        ax[i][j].text(0.5,-0.2, "Ground truth", size=12, ha="center", transform=ax[i][j].transAxes)
                    elif j == 2:
                        ax[i][j].text(0.5,-0.2, "Prediction", size=12, ha="center", transform=ax[i][j].transAxes)
                    elif j == 3:
                        ax[i][j].text(0.5,-0.2, "Uncertainty", size=12, ha="center", transform=ax[i][j].transAxes)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def display_pointcloud(xyz: np.ndarray, rgb: np.ndarray=None):
    """
    Displays a pointcloud using Open3D
    Args:
        xyz: (H*W, 3) ndarray containing the xyz-coordinates in each row
        rgb: (HxWx3) ndarray containing the rgb-values at each pixel location
    """
    pc_xyz = xyz.reshape(-1, 3)

    point_cloud_open3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc_xyz))

    if rgb is not None:
        pc_rgb = rgb.reshape(-1, 3)
        point_cloud_open3d.colors = o3d.utility.Vector3dVector(pc_rgb)

    visualizer = o3d.visualization.Visualizer()  # pylint: disable=no-member
    visualizer.create_window()
    visualizer.add_geometry(point_cloud_open3d)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    visualizer.add_geometry(origin)

    visualizer.get_render_option().background_color = (0, 0, 0)
    visualizer.get_render_option().point_size = 1
    visualizer.get_view_control().set_front([0, 0, -1])
    visualizer.get_view_control().set_up([0, -1, 0])

    visualizer.run()
    visualizer.destroy_window()


def depth_to_xyz(depth_image: np.ndarray, fx=3.4*1280/4.416, fy=3.4*720/2.484, cx=1280/2, cy=720/2, scale=0.3815*1e-3):
    """
    Converts depth image into a pointcloud
    Args:
        depth_image: (HxW) numpy array
        other: camera intrinsics
    Returns:
        xyz: (H*W,3) numpy array containing xyz-coordinates in each row
    """
    h,w = depth_image.shape
    
    x = np.linspace(0, w - 1, w).astype(np.int)
    y = np.linspace(0, h - 1, h).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    uv = np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])

    xyz = np.linalg.inv(K) @ uv * (depth_image*scale).flatten()
    return xyz.T

def plot_tensorboard_log(path):
    event_acc = EventAccumulator(path)
    event_acc.Reload()

    # Show all tags in the log file
    # print(event_acc.Tags())

    total_loss = eventacc_to_dict(event_acc.Scalars('losses/total_loss'))

    rmse = eventacc_to_dict(event_acc.Scalars('metrics/rmse'))
    logrmse = eventacc_to_dict(event_acc.Scalars('metrics/logrmse'))
    are = eventacc_to_dict(event_acc.Scalars('metrics/are'))
    sre = eventacc_to_dict(event_acc.Scalars('metrics/sre'))
    a1 = eventacc_to_dict(event_acc.Scalars('metrics/a1'))
    a2 = eventacc_to_dict(event_acc.Scalars('metrics/a2'))
    a3 = eventacc_to_dict(event_acc.Scalars('metrics/a3'))

    print("Best model (iteration, value)")
    print(f"RMSE:\t {get_min(rmse)[1]:.5f}")
    print(f"logRMSE: {get_min(logrmse)[1]:.5f}")
    print(f"ARD:\t {get_min(are)[1]:.5f}")
    print(f"SRD:\t {get_min(sre)[1]:.5f}")
    print(f"a1:\t {get_max(a1)[1]:.5f}")
    print(f"a2:\t {get_max(a2)[1]:.5f}")
    print(f"a3:\t {get_max(a3)[1]:.5f}")


def eventacc_to_dict(event_acc):
    event_dict = dict()
    for i in range(len(event_acc)):
        event_dict[event_acc[i][1]] = event_acc[i][2]

    return event_dict

def plot(x: dict, **kwargs):
    global_steps = list(x.keys())
    value = list(x.values())

    label = kwargs.get("label", None)
    smoothing = kwargs.get("smoothing", 0.0)
    color = kwargs.get("color", None)
    ylabel = kwargs.get("ylabel", "Loss")
    xlim = kwargs.get("xlim", None)
    ylim = kwargs.get("ylim", None)
    fontsize = kwargs.get("fontsize", 25)
    smooth_only = kwargs.get("smooth_only", False)
    alpha = kwargs.get("alpha", 1.0)
    legend_loc = kwargs.get("legend_loc", "upper right")

    smoothed = smooth(value, smoothing)
    
    if not smooth_only:
        plt.plot(global_steps, value, color=color, alpha=alpha*0.5)
    plt.plot(global_steps, smoothed, color=color, label=label, alpha=alpha)
    plt.xlabel("Iterations", size=fontsize)
    plt.ylabel(ylabel, size=fontsize)
    plt.xticks(size=fontsize)
    plt.yticks(size=fontsize)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if label is not None:
        plt.legend(loc=legend_loc, frameon=True, prop={'size':fontsize - 5})

def get_max(x: dict):
    value = max(x.values())
    key = max(x, key=x.get)

    return (key, value)

def get_min(x: dict):
    value = min(x.values())
    key = min(x, key=x.get)

    return (key, value)

def smooth(scalars: List[float], weight: float) -> List[float]:
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  
        smoothed.append(smoothed_val)                        
        last = smoothed_val                                  

    return smoothed