<#
keyframe extract using FFmpeg
#>

# get keyframe time using ffprobe
# ffprobe -i test.mp4 -v quiet -select_streams v -show_entries frame=pkt_pts_time,pict_type | Out-File ffprobe.txt -Encoding utf8

# keyframe extract
ffmpeg -i test.mp4 -vf "select=eq(pict_type\,I)" -vsync vfr -qscale:v 2 -f image2 ./%08d.jpg
