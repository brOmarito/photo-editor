"""
GPU Acceleration Utilities.
Detects and configures GPU acceleration for image processing via OpenCV CUDA.
Falls back gracefully to CPU processing when GPU is unavailable.
"""

import cv2
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("photo_editor.gpu")


class GPUAccelerator:
    """Manages GPU acceleration for image processing operations."""

    def __init__(self):
        self._gpu_available = False
        self._cuda_device_count = 0
        self._detect_gpu()

    def _detect_gpu(self):
        """
        Detect available CUDA-capable GPUs via OpenCV.
        Sets internal flags indicating whether GPU processing is available.
        """
        try:
            self._cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if self._cuda_device_count > 0:
                self._gpu_available = True
                cv2.cuda.setDevice(0)
                device_info = cv2.cuda.DeviceInfo(0)
                logger.info(
                    f"GPU detected: {device_info.name()} "
                    f"(CUDA devices: {self._cuda_device_count})"
                )
            else:
                logger.info("No CUDA-capable GPU detected. Using CPU processing.")
        except (cv2.error, AttributeError):
            logger.info("OpenCV CUDA support not available. Using CPU processing.")
            self._gpu_available = False

    @property
    def is_available(self) -> bool:
        """Returns True if GPU acceleration is available."""
        return self._gpu_available

    @property
    def device_count(self) -> int:
        """Returns the number of CUDA-capable devices detected."""
        return self._cuda_device_count

    def upload(self, image: np.ndarray) -> "cv2.cuda.GpuMat | np.ndarray":
        """
        Upload a numpy image array to GPU memory.
        Returns the original numpy array if GPU is not available.

        Args:
            image: Input image as numpy array (BGR format).

        Returns:
            GpuMat on GPU, or original numpy array on CPU.
        """
        if self._gpu_available:
            try:
                gpu_mat = cv2.cuda_GpuMat()
                gpu_mat.upload(image)
                return gpu_mat
            except cv2.error as e:
                logger.warning(f"GPU upload failed, falling back to CPU: {e}")
                return image
        return image

    def download(self, gpu_mat) -> np.ndarray:
        """
        Download an image from GPU memory back to CPU numpy array.
        If input is already a numpy array, returns it unchanged.

        Args:
            gpu_mat: GpuMat or numpy array.

        Returns:
            numpy array (BGR format).
        """
        if isinstance(gpu_mat, np.ndarray):
            return gpu_mat
        try:
            return gpu_mat.download()
        except (cv2.error, AttributeError) as e:
            logger.warning(f"GPU download failed: {e}")
            return gpu_mat

    def gaussian_blur(self, image: np.ndarray, ksize: tuple = (5, 5),
                      sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian blur with GPU acceleration if available.

        Args:
            image: Input image as numpy array.
            ksize: Kernel size as (width, height).
            sigma: Gaussian sigma value.

        Returns:
            Blurred image as numpy array.
        """
        if self._gpu_available:
            try:
                gpu_src = self.upload(image)
                # OpenCV CUDA Gaussian filter
                gaussian_filter = cv2.cuda.createGaussianFilter(
                    image.dtype, image.dtype, ksize, sigma
                )
                gpu_dst = gaussian_filter.apply(gpu_src)
                return self.download(gpu_dst)
            except (cv2.error, AttributeError) as e:
                logger.debug(f"GPU Gaussian blur failed, using CPU: {e}")

        return cv2.GaussianBlur(image, ksize, sigma)

    def resize(self, image: np.ndarray, size: tuple,
               interpolation: int = cv2.INTER_LANCZOS4) -> np.ndarray:
        """
        Resize image with GPU acceleration if available.

        Args:
            image: Input image as numpy array.
            size: Target size as (width, height).
            interpolation: Interpolation method.

        Returns:
            Resized image as numpy array.
        """
        if self._gpu_available:
            try:
                gpu_src = self.upload(image)
                gpu_dst = cv2.cuda.resize(gpu_src, size, interpolation=interpolation)
                return self.download(gpu_dst)
            except (cv2.error, AttributeError) as e:
                logger.debug(f"GPU resize failed, using CPU: {e}")

        return cv2.resize(image, size, interpolation=interpolation)


# Singleton GPU accelerator instance
_gpu_accelerator = None


def get_gpu_accelerator() -> GPUAccelerator:
    """
    Get the singleton GPUAccelerator instance.

    Returns:
        GPUAccelerator instance.
    """
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator()
    return _gpu_accelerator
