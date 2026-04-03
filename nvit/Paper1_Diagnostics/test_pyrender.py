import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
try:
    import pyrender
    import trimesh
    import numpy as np
    
    # Create simple scene
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.8, 0.3, 0.3, 1.0])
    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    
    scene = pyrender.Scene()
    scene.add(render_mesh)
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 2]
    scene.add(camera, pose=camera_pose)
    
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)
    
    r = pyrender.OffscreenRenderer(400, 400)
    color, depth = r.render(scene)
    print(f"Success! Rendered shape: {color.shape}")
    
except Exception as e:
    print(f"Failed! Error: {e}")
    import traceback
    traceback.print_exc()
