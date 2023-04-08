# imports for the HW
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


path = r'C:\Users\97250\PycharmProjects\anat_project\wet1\given_data\MilkyChance_StolenDance.mp4'


def video_to_frames(vid_path, start_second, end_second):
    """
    Load a video and return its frames from the wanted time range.
    :param vid_path: video file path.
    :param start_second: time of first frame to be taken from the
    video in seconds.
    :param end_second: time of last frame to be taken from the
    video in seconds.
    :return:
    frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C]
    containing the wanted video frames in BGR format.
    """
    cap = cv.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Cannot open video")
        exit()

    # Get the frame rate of the video
    fps = cap.get(cv.CAP_PROP_FPS)

    # Calculate the time of the first frame
    start_time = start_second * fps

    # Calculate the time of the last frame
    end_time = end_second * fps

    # Calculate the number of frames to be taken
    num_of_frames = int(end_time - start_time)
    # Create an empty array to store the frames
    frame_set = np.empty((num_of_frames, *cap.read()[1].shape), dtype=np.uint8)

    # Read the frames from the video
    for f in range(num_of_frames):
        cap.set(cv.CAP_PROP_POS_FRAMES, start_time + f)
        frame_set[f] = cap.read()[1]
    return frame_set


def gamma_correction(img, gamma):
    """
    Perform gamma correction on a grayscale image.
    :param img: An input grayscale image - ndarray of uint8 type.
    :param gamma: the gamma parameter for the correction.
    :return:
    gamma_img: An output grayscale image after gamma correction -
    uint8 ndarray of size [H x W x 1].
    """
    gamma_img = np.power(img, gamma)
    return gamma_img


def frames_mean(frame_set):
    """
    Calculate the mean frame out of a set of frames.
    :param frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C].
    :return:
    mean_frame: the average image calculated from the set,
    a 3D uint8 np array of size [H x W x C].
    """
    # Using np.mean as we can't use loops in this function
    mean_frame = np.mean(frame_set, axis=0)
    mean_frame = np.uint8(mean_frame)
    return mean_frame


def frames_median(frame_set):
    """
    Calculate the median frame out of a set of frames.
    :param frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C].
    :return:
    median_frame: the median image calculated from the set,
    a 3D uint8 np array of size [H x W x C].
    """
    median_frame = np.median(frame_set, axis=0)
    median_frame = np.uint8(median_frame)
    return median_frame


# Sample the 8th second's frame of the video
def Q1_1a(path, s_time, end_time):
    eight_frame = video_to_frames(path, s_time, end_time)
    frame = eight_frame[0]
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hist = plt.hist(gray.ravel(), 256, [0, 256], color='black', ec='black')
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('8th Second Frame')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.hist(gray.ravel(), 256, [0, 256], color='black', ec='black')
    plt.title('Histogram of the 8th second frame of the video')
    plt.axis('on')
    plt.show()


def Q1_1b(path, s_time, end_time):
    gamma = [0.5, 1.5]
    for g in gamma:
        eight_frame = video_to_frames(path, s_time, end_time)
        frame = eight_frame[0]
        # Gamma correction works only on grayscale images
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gamma_img = gamma_correction(gray, g)
        # hist = plt.hist(gray.ravel(), 256, [0, 256], color='black', ec='black')
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(gamma_img, cmap='gray')
        plt.title('Gamma correction with gamma = ' + str(g))
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.hist(gamma_img.ravel(), 256, [0, 256], color='black', ec='black')
        plt.title('Histogram of the gamma corrected image with gamma = ' + str(g))
        plt.axis('on')
        plt.show()


def Q1_1c(path, s_time, end_time):
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open video")
        exit()
    # Select a random frame from the video in the range 00:45:00-00:48:00
    frame_time = np.random.randint(s_time, end_time, size=1) / 100
    cap.set(cv.CAP_PROP_POS_MSEC, frame_time[0] * 1000)
    frame = cap.read()[1]
    cap.release()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 2, 1)
    plt.imshow(frame, cmap='gray')
    plt.title('Mean image of the video in 00:45 - 00:48')
    plt.axis('off')
    # Histogram of the sampled image of RGB values all together
    plt.subplot(2, 2, 2)
    plt.hist(frame[:, :, 0].ravel(), 256, [0, 256], color='red', ec='red', histtype='step')
    plt.title('Histogram of the Red color in sampled frame of the video')
    plt.subplot(2, 2, 3)
    plt.hist(frame[:, :, 1].ravel(), 256, [0, 256], color='green', ec='green', histtype='step')
    plt.title('Histogram of the Green color in sampled frame of the video')
    plt.subplot(2, 2, 4)
    plt.hist(frame[:, :, 2].ravel(), 256, [0, 256], color='blue', ec='blue', histtype='step')
    plt.title('Histogram of the Blue color in sampled frame of the video')
    plt.xlabel('Pixel value')
    plt.ylabel('Count')
    plt.show()


# Average in time – create the average image of the section
def Q1_1d(path, s_time, end_time):
    farme_set = video_to_frames(path, s_time, end_time)
    mean_frame = frames_mean(farme_set)
    frame = cv.cvtColor(mean_frame, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 2, 1)
    plt.imshow(frame, cmap='gray')
    plt.title('Mean image of the video in 00:45 - 00:48')
    plt.axis('off')
    # Histogram of the sampled image of RGB values all together
    plt.subplot(2, 2, 2)
    plt.hist(frame[:, :, 0].ravel(), 256, [0, 256], color='red', ec='red', histtype='step')
    plt.title('Histogram of the Red color in Mean frame of the video')
    plt.subplot(2, 2, 3)
    plt.hist(frame[:, :, 1].ravel(), 256, [0, 256], color='green', ec='green', histtype='step')
    plt.title('Histogram of the Green color Mean frame of the video')
    plt.subplot(2, 2, 4)
    plt.hist(frame[:, :, 2].ravel(), 256, [0, 256], color='blue', ec='blue', histtype='step')
    plt.title('Histogram of the Blue color Mean frame of the video')
    plt.xlabel('Pixel value')
    plt.ylabel('Count')
    plt.show()


# Median in time – create the median image of the section
def Q1_2d(path, s_time, end_time):
    farme_set = video_to_frames(path, s_time, end_time)
    median_frame = frames_median(farme_set)
    frame = cv.cvtColor(median_frame, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 4))
    plt.subplot(2, 2, 1)
    plt.imshow(frame, cmap='gray')
    plt.title('Mean image of the video in 00:45 - 00:48')
    plt.axis('off')
    # Histogram of the sampled image of RGB values all together
    plt.subplot(2, 2, 2)
    plt.hist(frame[:, :, 0].ravel(), 256, [0, 256], color='red', ec='red', histtype='step')
    plt.title('Histogram of the Red color in Median frame of the video')
    plt.subplot(2, 2, 3)
    plt.hist(frame[:, :, 1].ravel(), 256, [0, 256], color='green', ec='green', histtype='step')
    plt.title('Histogram of the Green color in Median frame of the video')
    plt.subplot(2, 2, 4)
    plt.hist(frame[:, :, 2].ravel(), 256, [0, 256], color='blue', ec='blue', histtype='step')
    plt.title('Histogram of the Blue color in Median frame of the video')
    plt.xlabel('Pixel value')
    plt.ylabel('Count')
    plt.show()


if __name__ == '__main__':
    # s_time = 8
    # end_time = 9
    # Q1_1a(path, s_time, end_time)
    # Q1_1b(path, s_time, end_time)
    s_time = 4500
    end_time = 4800
    # Q1_1c(path, s_time, end_time)
    s_time = 45
    end_time = 48
    # Q1_1d(path, s_time, end_time)
    # Q1_2d(path, s_time, end_time)