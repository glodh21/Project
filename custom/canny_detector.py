import numpy as np
import cv2
from collections import deque


class CannyEdgeDetector:

    def __init__(
        self,
        sigma=1.4,
        gaussian_size=5
    ):

        self.sigma = sigma

        self.gaussian_size = (
            gaussian_size
            if gaussian_size % 2 == 1
            else gaussian_size + 1
        )

        self.intermediate = {}

    # --------------------------------------------------
    # MAIN DETECTION FUNCTION
    # --------------------------------------------------

    def detect(
        self,
        image,
        low_threshold=50,
        high_threshold=150
    ):

        low_threshold = low_threshold / 255.0
        high_threshold = high_threshold / 255.0

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

        # Convert to float
        if image.dtype == np.uint8:
            img_float = image.astype(
                np.float32
            ) / 255.0
        else:
            img_float = image.copy()

        # --------------------------------------------------
        # STEP 1: GAUSSIAN SMOOTHING
        # --------------------------------------------------

        smoothed = self.gaussian_smooth(
            img_float
        )

        # --------------------------------------------------
        # STEP 2: GRADIENTS
        # --------------------------------------------------

        gradient_mag, gradient_dir = (
            self.compute_gradients(smoothed)
        )

        # --------------------------------------------------
        # STEP 3: NON-MAXIMUM SUPPRESSION
        # --------------------------------------------------

        suppressed = self.non_maximum_suppression(
            gradient_mag,
            gradient_dir
        )

        # --------------------------------------------------
        # STEP 4: HYSTERESIS THRESHOLDING
        # --------------------------------------------------

        edges = self.hysteresis_thresholding(
            suppressed,
            low_threshold,
            high_threshold
        )

        return edges

    # --------------------------------------------------
    # GAUSSIAN KERNEL
    # --------------------------------------------------

    def gaussian_kernel(
        self,
        size,
        sigma
    ):

        center = size // 2

        x = np.arange(size) - center

        kernel_1d = np.exp(
            -(x ** 2) / (2 * sigma ** 2)
        )

        kernel_2d = np.outer(
            kernel_1d,
            kernel_1d
        )

        kernel_2d = (
            kernel_2d / np.sum(kernel_2d)
        )

        return kernel_2d

    # --------------------------------------------------
    # GAUSSIAN SMOOTHING
    # --------------------------------------------------

    def gaussian_smooth(
        self,
        image
    ):

        kernel = self.gaussian_kernel(
            self.gaussian_size,
            self.sigma
        )

        smoothed = cv2.filter2D(
            image,
            -1,
            kernel,
            borderType=cv2.BORDER_REPLICATE
        )

        return smoothed

    # --------------------------------------------------
    # SOBEL KERNELS
    # --------------------------------------------------

    def sobel_kernel(self):

        gx = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ],
            dtype=np.float32
        )

        gy = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ],
            dtype=np.float32
        )

        return gx, gy

    # --------------------------------------------------
    # COMPUTE GRADIENTS
    # --------------------------------------------------

    def compute_gradients(
        self,
        image
    ):

        gx_kernel, gy_kernel = (
            self.sobel_kernel()
        )

        gx = cv2.filter2D(
            image,
            -1,
            gx_kernel,
            borderType=cv2.BORDER_REPLICATE
        )

        gy = cv2.filter2D(
            image,
            -1,
            gy_kernel,
            borderType=cv2.BORDER_REPLICATE
        )

        magnitude = np.sqrt(
            gx ** 2 + gy ** 2
        )

        magnitude = (
            magnitude / (np.max(magnitude) + 1e-8)
        )

        direction = np.arctan2(
            gy,
            gx
        ) * 180.0 / np.pi

        direction[direction < 0] += 180

        return magnitude, direction

    # --------------------------------------------------
    # NON-MAXIMUM SUPPRESSION
    # --------------------------------------------------

    def non_maximum_suppression(
        self,
        magnitude,
        direction
    ):

        suppressed = np.zeros_like(
            magnitude
        )

        for i in range(1, magnitude.shape[0] - 1):

            for j in range(1, magnitude.shape[1] - 1):

                angle = direction[i, j]

                q = 255
                r = 255

                if (
                    (0 <= angle < 22.5)
                    or
                    (157.5 <= angle <= 180)
                ):

                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]

                elif 22.5 <= angle < 67.5:

                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]

                elif 67.5 <= angle < 112.5:

                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]

                elif 112.5 <= angle < 157.5:

                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]

                if (
                    magnitude[i, j] >= q
                    and
                    magnitude[i, j] >= r
                ):

                    suppressed[i, j] = magnitude[i, j]

        return suppressed

    # --------------------------------------------------
    # HYSTERESIS THRESHOLDING
    # --------------------------------------------------

    def hysteresis_thresholding(
        self,
        suppressed,
        low_thresh_val,
        high_thresh_val
    ):

        edges = np.zeros_like(
            suppressed,
            dtype=np.uint8
        )

        strong = suppressed >= high_thresh_val

        weak = (
            (suppressed >= low_thresh_val)
            &
            (suppressed < high_thresh_val)
        )

        edges[strong] = 255

        visited = strong.copy()

        queue = deque(
            list(zip(*np.where(strong)))
        )

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        height, width = edges.shape

        while queue:

            x, y = queue.popleft()

            for dx, dy in neighbors:

                nx, ny = x + dx, y + dy

                if (
                    0 <= nx < height
                    and
                    0 <= ny < width
                ):

                    if (
                        not visited[nx, ny]
                        and weak[nx, ny]
                    ):

                        edges[nx, ny] = 255

                        visited[nx, ny] = True

                        queue.append((nx, ny))

        return edges