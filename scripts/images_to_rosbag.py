
''' 
Images extracted at a rate of approximately 5FPS have been hand-labeled.
To exploit the timing capabilities of ROS, we want to be able to 
perform inference on a rosbag (as a video stream) rather than a 
sequence of individual image frames.
Therefore in order to facilitate evaluation against our desired
metrics we wish to create a new rosbag with only the labeled frames,
which is the purpose of this script.
We combine both color and thermal into the same bag, for convenience.
'''

import rospy
import cv2
from sensor_msgs.msg import Image
import rosbag
import os
from cv_bridge import CvBridge

# Define file paths
image_path = '/home/ed/Data/frames/images/'
bag_out_path = '/home/ed/Data/frames/rosbag/'
seqs = []

topic_names = {'color': 'color', 'thermal': 'thermal'}
encodings = {'color': 'bgr8', 'thermal': 'mono8'}

br = CvBridge()

def filename2time(filename):
    sec, nsec = filename.split('_')[1:3]
    return rospy.Time(int(sec), int(nsec))

# Work out the sequences to read
for root, dirs, files in os.walk(image_path + 'color/'):
    if not dirs:
        seqs.append(root.split('/')[-1])

# Go through each sequence
for seq in sorted(seqs):
    print ('Processing', seq)
    cv_images = {'color': [], 'thermal': []}
    times = {'color': [], 'thermal': []}
    for image_type in cv_images.keys():
        # Read in the color and thermal frames and times to their respective buffers
        for root, dirs, files in os.walk(image_path + image_type + '/' + seq):
            for file in sorted(files):
                if file.endswith('.jpg'):
                    im = cv2.imread(os.path.join(root, file))
                    if 'thermal' in file:
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    cv_images[image_type].append(im)
                    time = filename2time(file)
                    times[image_type].append(time)

    with rosbag.Bag(os.path.join(bag_out_path, seq + '.bag'), 'w') as outbag:
        print('Writing bag:', os.path.join(bag_out_path, seq + '.bag'))
        for image_type in cv_images.keys():
            topic = topic_names[image_type]
            for im, t in zip(cv_images[image_type], times[image_type]):
                msg = br.cv2_to_imgmsg(im, encoding=encodings[image_type])
                msg.header.stamp = t
                outbag.write(topic, msg, t)