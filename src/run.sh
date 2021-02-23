# Haar cascade
# python test.py --vflip=True --display_video=True --detect=True

# Frame difference motion detector
# python test.py --vflip=True --display_video=True --detect_motion=True
# python test.py --vflip=True --display_video=True --detect=True
# python test.py --vflip=True --display_video=True --detect_motion=True --frame_width=640 --frame_height=480
python test.py --vflip=True --display_video=True --detect_motion=True --frame_width=800 --frame_height=600
# python test.py --vflip=True --display_video=True --detect_motion=True --frame_width=800 --frame_height=600

# Test on Mac with default res
# python test.py --hflip=True --display_video=True --detect_motion=True

# On MacBook Air M1 with default res
# Having to use compatability mode at present - hopefully this will go native soon.
# arch -x86_64 python test.py --hflip=True --display_video=True --detect_motion=True
