# RealSense FAQ

## Python

### 1. Frame didn't arrive within 5000

```log
发生异常: RuntimeError
Frame didn't arrive within 5000
  File "/media/supervisor/windowsd/Code/healthpose-vitpose/health/camera.py", line 44, in update
    frames = self.pipeline.wait_for_frames()
  File "/home/supervisor/miniconda3/envs/mediapipe/lib/python3.10/threading.py", line 953, in run
    self._target(*self._args, **self._kwargs)
  File "/home/supervisor/miniconda3/envs/mediapipe/lib/python3.10/threading.py", line 1016, in _bootstrap_inner
    self.run()
  File "/home/supervisor/miniconda3/envs/mediapipe/lib/python3.10/threading.py", line 973, in _bootstrap
    self._bootstrap_inner()
RuntimeError: Frame didn't arrive within 5000
```

在 `pipline.start()` 后，调用 `device.hardware_reset()` 解决。

```python
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    profile = pipeline.start(config)
    device = profile.get_device()
    device.hardware_reset()
```

### 2.

```log

```
