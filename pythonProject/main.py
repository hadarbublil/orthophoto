import itertools
import typing
from functools import cache
from typing import Iterator, Iterable

import cv2
import os

import loguru
from skimage.metrics import structural_similarity as ssim
import numpy

is_processing = False

Frame = cv2.typing.MatLike


def iter_frames(video_captures: cv2.VideoCapture, frame_interval: int = 3) -> Iterator[Frame]:
    for i in range(int(video_captures.get(cv2.CAP_PROP_FRAME_COUNT))):
        success, frame = video_captures.read()
        if not success:
            break
        if i % frame_interval == 0:
            yield frame


def save_frames_to_folder(folder_path: str, frames: Iterable[Frame]):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for idx, frame in enumerate(frames):
        frame_filename = os.path.join(folder_path, f"frame_{idx}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f'Saved: {frame_filename}')


def is_blurry(frame: Frame, threshold=1000) -> int:
    variance = calc_blurr(frame)
    loguru.logger.info(f"frame blurr is {variance} expecting {threshold}")
    return variance < threshold


def calc_blurr(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance


def sim1(frame1: Frame, frame2: Frame) -> float:
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    return ssim(gray1, gray2)


def adjust_lighting(frame: Frame) -> Frame:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    adjusted_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return adjusted_frame


def sharpen_frame(frame: Frame) -> Frame:
    kernel = numpy.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
    image_sharp = cv2.filter2D(src=frame, ddepth=-1, kernel=kernel)
    return image_sharp


def sim2(f1, f2):
    f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(f1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(f2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return len(good_matches) / min(len(keypoints1), len(keypoints2))


def sim3(img1, img2):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = [m for m in matches if m.distance < 50]  # distance threshold for good matches

    similarity_percentage = (len(good_matches) / min(len(kp1), len(kp2)))

    return similarity_percentage


def reduce_noise_gaussian(frame: Frame, kernel_size: tuple[int, int] = (5, 5), sigma: int = 0) -> Frame:
    reduced_noise_frame = cv2.GaussianBlur(frame, kernel_size, sigma)
    return reduced_noise_frame


def process_frames(frames: Iterable[Frame]) -> Iterator[Frame]:
    for frame in frames:
        loguru.logger.info(f"blurr before processed {calc_blurr(frame)}")
        adjusted_frame = adjust_lighting(frame)  # Adjust lighting
        sharpened_frame = sharpen_frame(adjusted_frame)  # Sharpen the adjusted frame
        reduced_noise_frame = reduce_noise_gaussian(sharpened_frame)
        yield reduced_noise_frame


def frame_capture(
        v_path: str,
        sample_rate: int,
        blurr_threshold: int,
        similarity_threshold: float,
        min_group_size: int = 2,
        output_dir='filtered_group_frames'
) -> str | None:
    global is_processing
    is_processing = True
    video = cv2.VideoCapture(v_path)
    frames = iter_frames(video, sample_rate)
    processed_frames = process_frames(frames)

    filtered_frames = filter(
        lambda f: is_blurry(f, blurr_threshold),
        processed_frames
    )

    grouped = group_by(
        filtered_frames,
        lambda f1, f2: is_similar(f1, f2, similarity_threshold)
    )

    for count, group in enumerate(grouped):
        if not is_processing:
            return None

        loguru.logger.info(f"group size is {len(group)}")
        if len(group) < min_group_size:
            continue

        folder_path = os.path.join(output_dir, f"group_{count}")
        save_frames_to_folder(folder_path, group)
        break  # todo: currently stopping after first group

    video.release()
    return output_dir


def is_similar(frame1: Frame, frame2: Frame, similarity_threshold: float = 0.2) -> bool:
    sim = sim3(frame1, frame2)
    loguru.logger.info(f"similarity is {sim} expecting {similarity_threshold}")
    return sim > similarity_threshold


T = typing.TypeVar("T")


def group_by(iterable: Iterable[T], condition_fn: typing.Callable[[T, T], bool]) -> Iterator[list[T]]:
    it = iter(iterable)
    group = [next(it)]

    for item in it:
        if condition_fn(group[-1], item):
            group.append(item)
        else:
            yield group
            group = [item]

    yield group


def load_frames_from_folder(folder_path: str) -> list[Frame]:
    frames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            frame = cv2.imread(file_path)
            if frame is not None:
                frames.append(frame)
    return frames


#  docker run -ti --rm -v c:/Users/User/PycharmProjects/orthomosaic/pythonProject:/datasets opendronemap/odm --project-path /datasets --end-with odm_orthophoto --orthophoto-resolution 1 filtered_group_frames

if __name__ == '__main__':
    video_path = r"C:\Users\User\Desktop\DemoVideofromCrane.mp4"
    frame_capture(
        video_path,
        sample_rate=15,
        blurr_threshold=400,
        similarity_threshold=0.29,
        min_group_size=10,
    )

    base_path = r"C:\Users\User\PycharmProjects\orthomosaic\pythonProject\filtered_group_frames"
    # od.run_odm_for_folders(base_path)
