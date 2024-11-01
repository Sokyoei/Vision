"""
https://github.com/keplerlab/Katna
https://blog.csdn.net/qq_15969343/article/details/124157138
"""

from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter


def main():
    vd = Video()
    no_of_frames_to_returned = 12
    disk_writer = KeyFrameDiskWriter(location=r'keyframe_extract_katna')
    video_file_path = r'test.mp4'
    vd.extract_video_keyframes(no_of_frames=no_of_frames_to_returned, file_path=video_file_path, writer=disk_writer)


if __name__ == "__main__":
    main()
