import open3d as o3d

# 加载点云
pcd = o3d.io.read_point_cloud(r"F:\PythonWork\4.opencv\3.open3d\office\office2.pcd")
# Alpha shapes
alpha = 0.1
print(f"alpha={alpha:.3f}")
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
mesh.compute_vertex_normals()
# 可视化重建结果
o3d.visualization.draw_geometries([mesh], window_name="点云重建",
                                  width=800,
                                  height=600,
                                  mesh_show_back_face=True)