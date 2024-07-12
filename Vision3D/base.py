import gzip
import os
from pathlib import Path

import open3d as o3d
import requests
from tqdm import tqdm


def download_data(url: str):
    response = requests.get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (raKHTML, like Gecko)"
            " Chrome/109.0.0.0 Safari/537.36 Edg/109.0.1518.78 "
        },
        timeout=10,
    )
    file_name = url.split("/")[-1]
    total = int(response.headers.get("Content-Length", 0))
    with open(file_name, "wb") as f, tqdm(
        desc=file_name, total=total, unit="iB", unit_scale=True, unit_divisor=1024
    ) as t:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            t.update(size)
    response.close()


def read_write_pcl():
    # download and extra data
    url = "https://graphics.stanford.edu/data/3Dscanrep/xyzrgb/xyzrgb_dragon.ply.gz"
    gz_file_name = "xyzrgb_dragon.ply.gz"
    file_name = "xyzrgb_dragon.ply"
    if not Path(file_name).exists():
        download_data(url)
        gz = gzip.open(gz_file_name)
        with open(file_name, "wb") as f:
            f.write(gz.read())
        gz.close()
        os.remove(gz_file_name)

    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(file_name)
    print(pcd)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.01, 30))

    o3d.visualization.draw_geometries([pcd])

    # o3d.io.write_point_cloud()


def main():
    read_write_pcl()


if __name__ == "__main__":
    main()
