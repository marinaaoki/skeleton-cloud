import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_point_cloud(joints, lines):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(joints)
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(joints),
        lines=o3d.utility.Vector2iVector(lines),
    )

    o3d.visualization.draw_geometries([pcd])

def plot_skeleton(joints, lines):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, joint in enumerate(joints):
        ax.scatter(joint[0], joint[1], joint[2], c='r', marker='o')
        ax.text(joint[0], joint[1], joint[2], '%s' % (str(i+1)), size=10, zorder=1, color='k')

    for line in lines:
        ax.plot(joints[line, 0], joints[line, 1], joints[line, 2], c='b')


    ax.view_init(elev=-50, azim=175, roll=-90)
    ax.invert_zaxis()

    # do not display the grid lines or axis numbers
    ax.grid(False)
    ax.axis('off')

    plt.show()

