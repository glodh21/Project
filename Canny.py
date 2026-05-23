import numpy as np
import cv2
from collections import deque

class CannyEdgeDetector:
    def __init__(self, sigma=1.4, low_threshold=0.04, high_threshold=0.10, gaussian_size=5):
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.gaussian_size = gaussian_size if gaussian_size % 2 == 1 else gaussian_size + 1
        self.intermediate = {}
    
    def detect(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        if image.dtype == np.uint8:
            img_float = image.astype(np.float32) / 255.0
        else:
            img_float = image.copy()
        
        smoothed = self.gaussian_smooth(img_float)
        self.intermediate['smoothed'] = smoothed
        
        gradient_mag, gradient_dir = self.compute_gradients(smoothed)
        self.intermediate['gradient_magnitude'] = gradient_mag
        self.intermediate['gradient_direction'] = gradient_dir
        
        suppressed = self.non_maximum_suppression(gradient_mag, gradient_dir)
        self.intermediate['suppressed'] = suppressed
        
        edges = self.hysteresis_thresholding(suppressed)
        self.intermediate['edges'] = edges
        
        return edges
    
    def gaussian_kernel(self, size, sigma):
        center = size // 2
        x = np.arange(size) - center
        kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
        kernel_2d = np.outer(kernel_1d, kernel_1d)
        return kernel_2d / np.sum(kernel_2d)
    
    def gaussian_smooth(self, image):
        kernel = self.gaussian_kernel(self.gaussian_size, self.sigma)
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    
    def sobel_kernel(self):
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        return gx, gy
    
    def compute_gradients(self, image):
        gx_kernel, gy_kernel = self.sobel_kernel()
        gx = cv2.filter2D(image, -1, gx_kernel, borderType=cv2.BORDER_REPLICATE)
        gy = cv2.filter2D(image, -1, gy_kernel, borderType=cv2.BORDER_REPLICATE)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        magnitude = magnitude / (np.max(magnitude) + 1e-8)
        
        direction = np.arctan2(gy, gx) * 180.0 / np.pi
        direction[direction < 0] += 180
        
        return magnitude, direction
    
    def non_maximum_suppression(self, magnitude, direction):
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        
        idx_0   = ((direction >= 0) & (direction < 22.5)) | ((direction >= 157.5) & (direction < 180))
        idx_45  = (direction >= 22.5) & (direction < 67.5)
        idx_90  = (direction >= 67.5) & (direction < 112.5)
        idx_135 = (direction >= 112.5) & (direction < 157.5)
        
        mag_core = magnitude[1:-1, 1:-1]
        
        l_neighbor = magnitude[1:-1, 0:-2]
        r_neighbor = magnitude[1:-1, 2:]
        t_neighbor = magnitude[0:-2, 1:-1]
        b_neighbor = magnitude[2:, 1:-1]
        
        tl_neighbor = magnitude[0:-2, 0:-2]
        br_neighbor = magnitude[2:, 2:]
        tr_neighbor = magnitude[0:-2, 2:]
        bl_neighbor = magnitude[2:, 0:-2]
        
        mask_0   = idx_0[1:-1, 1:-1]   & (mag_core >= l_neighbor) & (mag_core >= r_neighbor)
        mask_45  = idx_45[1:-1, 1:-1]  & (mag_core >= tl_neighbor) & (mag_core >= br_neighbor)
        mask_90  = idx_90[1:-1, 1:-1]  & (mag_core >= t_neighbor) & (mag_core >= b_neighbor)
        mask_135 = idx_135[1:-1, 1:-1] & (mag_core >= tr_neighbor) & (mag_core >= bl_neighbor)
        
        combined_mask = mask_0 | mask_45 | mask_90 | mask_135
        suppressed[1:-1, 1:-1][combined_mask] = mag_core[combined_mask]
        
        return suppressed
    
    def hysteresis_thresholding(self, suppressed):
        high_thresh_val = self.high_threshold
        low_thresh_val = self.low_threshold
        
        edges = np.zeros_like(suppressed, dtype=np.uint8)
        
        strong = suppressed >= high_thresh_val
        weak = (suppressed >= low_thresh_val) & (suppressed < high_thresh_val)
        
        edges[strong] = 255
        
        visited = strong.copy()
        strong_positions = list(zip(*np.where(strong)))
        queue = deque(strong_positions)
        
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
        
        height, width = edges.shape
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in neighbors:
                nx, ny = x + dx, y + dy
                if 0 <= nx < height and 0 <= ny < width:
                    if not visited[nx, ny] and weak[nx, ny]:
                        edges[nx, ny] = 255
                        visited[nx, ny] = True
                        queue.append((nx, ny))
        
        return edges

    def get_intermediate_results(self):
        return self.intermediate

if __name__ == "__main__":
    import os
    import time
    
    image = cv2.imread('lana.png')
    if image is None:
        print("Error: Could not read image 'lana.png'")
        exit()
        
    os.makedirs('Outputs', exist_ok=True)
    
    detector = CannyEdgeDetector(sigma=1.4, low_threshold=0.04, high_threshold=0.10, gaussian_size=5)
    
    t0 = time.time()
    edges = detector.detect(image)
    print(f"Production-grade pipeline completed in: {time.time() - t0:.4f} seconds!")
    
    cv2.imwrite('Outputs/newedges.jpg', edges)