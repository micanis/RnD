import zarr
import matplotlib.pyplot as plt
import cv2

store = zarr.ZipStore("data/processed/image/dual_fisheye/test2.zarr.zip", mode="r")
root = zarr.group(store=store)

frame = root["right"][0]
plt.imshow(frame)
plt.axis("off")
plt.show()

cv2.imwrite("tmp/vis.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

store.close()
