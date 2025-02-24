import os
from datetime import datetime

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

from PIL import Image
from rosbags.image import message_to_cvimage


with Reader("/home/charlie/Desktop/PerpetuaLocobotDetector/rosbag2_2025_02_21-15_46_30") as reader:
    temp = reader.topics
    image_msg = temp["/locobot/camera/color/image_raw"].connections

    def ros2_time_to_string(nanoseconds: int):
        unix_timestamp = nanoseconds * 1e-9
        return datetime.fromtimestamp(unix_timestamp).strftime("%Y-%m-%d_%H-%M-%S")

    for connection, timestamp, rawdata in reader.messages(image_msg):
        msg = deserialize_cdr(rawdata, connection.msgtype)

        img = message_to_cvimage(msg)

        result = Image.fromarray(img)
        os.makedirs("./converted", exist_ok=True)
        result.save(f"/home/charlie/Desktop/PerpetuaLocobotDetector/converted/{ros2_time_to_string(timestamp)}.jpg")
