"""GPU acceleration utilities using CuPy (CUDA) or PyTorch (MPS for Mac)."""
import warnings
import platform
import sys
from typing import Optional, Tuple

import numpy as np
from scipy import sparse

# Try CuPy for CUDA (NVIDIA GPUs)
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    cusp = None

# Try PyTorch for MPS (Mac GPUs) or CUDA fallback
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = None
    torch = None


def diagnose_gpu_support() -> dict:
    """
    Diagnose GPU support availability and provide helpful information.
    
    Returns:
        Dictionary with diagnostic information
    """
    diagnostics = {
        'platform': platform.system(),
        'is_mac': platform.system() == 'Darwin',
        'python_version': sys.version,
        'cupy_available': CUPY_AVAILABLE,
        'torch_available': TORCH_AVAILABLE,
        'torch_version': TORCH_VERSION,
        'gpu_backends': []
    }
    
    # Check CuPy/CUDA
    if CUPY_AVAILABLE:
        try:
            _ = cp.array([1, 2, 3])
            device = cp.cuda.Device(0)
            device.use()
            diagnostics['gpu_backends'].append('CUDA (CuPy)')
        except Exception as e:
            diagnostics['cupy_error'] = str(e)
    
    # Check PyTorch MPS (Mac)
    if TORCH_AVAILABLE and diagnostics['is_mac']:
        try:
            if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
                if torch.backends.mps.is_available():
                    diagnostics['gpu_backends'].append('MPS (Mac)')
                else:
                    diagnostics['mps_unavailable_reason'] = 'MPS backend not available (requires macOS 12.3+ and Apple Silicon)'
            else:
                diagnostics['mps_unavailable_reason'] = f'PyTorch {TORCH_VERSION} does not support MPS (requires PyTorch 1.12+)'
        except Exception as e:
            diagnostics['mps_error'] = str(e)
    
    # Check PyTorch CUDA
    if TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                diagnostics['gpu_backends'].append('CUDA (PyTorch)')
        except Exception:
            pass
    
    return diagnostics


class GPUAccelerator:
    """GPU accelerator with automatic CPU fallback. Supports CUDA (NVIDIA) and MPS (Mac)."""
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize GPU accelerator.
        
        Args:
            use_gpu: Whether to attempt GPU acceleration
        """
        self.use_gpu = False
        self.device_type = None  # 'cuda', 'mps', or None
        self.device = None
        self.backend = None  # 'cupy' or 'torch'
        
        if not use_gpu:
            return
        
        # Detect platform and available GPU backends
        is_mac = platform.system() == 'Darwin'
        
        # Try CuPy (CUDA) first on non-Mac systems
        if not is_mac and CUPY_AVAILABLE:
            try:
                _ = cp.array([1, 2, 3])
                self.device = cp.cuda.Device(0)
                self.device.use()
                self.use_gpu = True
                self.device_type = 'cuda'
                self.backend = 'cupy'
                return
            except Exception as e:
                # CuPy failed, will try PyTorch CUDA as fallback
                if use_gpu:
                    warnings.warn(f"CuPy CUDA initialization failed: {e}. Trying PyTorch CUDA...")
                pass
        
        # Try PyTorch MPS (Mac GPU)
        if TORCH_AVAILABLE and is_mac:
            try:
                # Check if MPS backend is available (requires PyTorch 1.12+ and macOS 12.3+)
                if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
                    if torch.backends.mps.is_available():
                        self.device = torch.device('mps')
                        # Test with a small operation
                        test_tensor = torch.tensor([1.0, 2.0, 3.0], device=self.device)
                        _ = test_tensor * 2
                        self.use_gpu = True
                        self.device_type = 'mps'
                        self.backend = 'torch'
                        return
                    else:
                        warnings.warn("MPS backend is not available. This requires macOS 12.3+ and Apple Silicon (M1/M2/M3). Falling back to CPU.")
                else:
                    warnings.warn("PyTorch MPS backend not available. Please upgrade PyTorch to version 1.12+ for Mac GPU support. Falling back to CPU.")
            except Exception as e:
                warnings.warn(f"MPS GPU acceleration failed: {e}. Falling back to CPU.")
        
        # Try PyTorch CUDA as fallback (if available) - works on Windows/Linux
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                    test_tensor = torch.tensor([1.0, 2.0, 3.0], device=self.device)
                    _ = test_tensor * 2
                    self.use_gpu = True
                    self.device_type = 'cuda'
                    self.backend = 'torch'
                    return
                else:
                    if use_gpu and not is_mac:
                        warnings.warn("PyTorch CUDA is not available. Make sure CUDA drivers and PyTorch with CUDA support are installed.")
            except RuntimeError as e:
                # Catch CUDA capability errors (e.g., sm_120 not supported)
                error_msg = str(e)
                if "CUDA capability" in error_msg and "not compatible" in error_msg:
                    if use_gpu:
                        warnings.warn(f"GPU architecture not supported by current PyTorch: {error_msg}. "
                                    "Try installing PyTorch nightly build for newer GPU support: "
                                    "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124")
                else:
                    if use_gpu:
                        warnings.warn(f"PyTorch CUDA initialization failed: {e}")
            except Exception as e:
                if use_gpu:
                    warnings.warn(f"PyTorch CUDA initialization failed: {e}")
        
        # No GPU available
        if use_gpu:
            # Provide helpful diagnostics
            if not is_mac:
                # Windows/Linux - check what's available
                if not CUPY_AVAILABLE and not TORCH_AVAILABLE:
                    warnings.warn("GPU acceleration requested but neither CuPy nor PyTorch is installed. "
                                "Install CuPy with: pip install cupy-cuda12x (or appropriate CUDA version)")
                elif CUPY_AVAILABLE and not TORCH_AVAILABLE:
                    warnings.warn("GPU acceleration requested but CuPy CUDA failed and PyTorch is not installed. "
                                "Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
                elif TORCH_AVAILABLE and not torch.cuda.is_available():
                    warnings.warn("GPU acceleration requested but PyTorch reports CUDA is not available. "
                                "Check CUDA installation and PyTorch CUDA compatibility.")
                else:
                    warnings.warn("GPU acceleration requested but no GPU backend available. Falling back to CPU.")
            else:
                warnings.warn("GPU acceleration requested but no GPU backend available. Falling back to CPU.")
    
    def to_gpu(self, arr: np.ndarray):
        """Convert numpy array to GPU array."""
        if not self.use_gpu:
            return arr
        
        if self.backend == 'cupy':
            gpu_arr = cp.asarray(arr)
            # Verify data is on GPU
            if not isinstance(gpu_arr, cp.ndarray):
                raise RuntimeError("Failed to convert array to CuPy GPU array")
            return gpu_arr
        elif self.backend == 'torch':
            if self.device is None:
                raise RuntimeError("PyTorch device is None - GPU not properly initialized")
            # Convert to tensor and move to device
            gpu_tensor = torch.from_numpy(arr).to(self.device)
            # Verify tensor is on correct device
            if gpu_tensor.device != self.device:
                raise RuntimeError(f"Tensor not on expected device: {gpu_tensor.device} != {self.device}")
            return gpu_tensor
        return arr
    
    def to_cpu(self, arr) -> np.ndarray:
        """Convert GPU array back to CPU numpy array."""
        if not self.use_gpu:
            return arr
        
        if self.backend == 'cupy' and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        elif self.backend == 'torch' and isinstance(arr, torch.Tensor):
            return arr.cpu().numpy()
        return arr
    
    def sparse_to_gpu(self, sp_matrix: sparse.spmatrix) -> 'cusp.spmatrix':
        """Convert scipy sparse matrix to CuPy sparse matrix."""
        if self.use_gpu:
            if isinstance(sp_matrix, sparse.csr_matrix):
                return cusp.csr_matrix(sp_matrix)
            elif isinstance(sp_matrix, sparse.csc_matrix):
                return cusp.csc_matrix(sp_matrix)
            elif isinstance(sp_matrix, sparse.coo_matrix):
                return cusp.coo_matrix(sp_matrix)
        return sp_matrix
    
    def sparse_to_cpu(self, sp_matrix) -> sparse.spmatrix:
        """Convert CuPy sparse matrix back to scipy sparse matrix."""
        if self.use_gpu and hasattr(sp_matrix, 'get'):
            # CuPy sparse matrix
            return sp_matrix.get()
        return sp_matrix
    
    def svd(self, matrix: np.ndarray, n_components: int, 
            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Truncated SVD on GPU or CPU.
        
        Args:
            matrix: Input matrix (n_samples, n_features)
            n_components: Number of components
            random_state: Random seed
            
        Returns:
            (U, S, Vt) where U @ diag(S) @ Vt approximates the input
        """
        if self.use_gpu:
            try:
                gpu_matrix = self.to_gpu(matrix)
                
                # Verify data is on GPU
                if self.backend == 'cupy':
                    if not isinstance(gpu_matrix, cp.ndarray):
                        raise RuntimeError("Matrix not on GPU (CuPy)")
                elif self.backend == 'torch':
                    if not isinstance(gpu_matrix, torch.Tensor) or gpu_matrix.device.type != self.device.type:
                        raise RuntimeError(f"Matrix not on GPU (PyTorch): device={gpu_matrix.device if isinstance(gpu_matrix, torch.Tensor) else 'not tensor'}")
                
                if self.backend == 'cupy':
                    # CuPy SVD
                    U, S, Vt = cp.linalg.svd(gpu_matrix, full_matrices=False)
                elif self.backend == 'torch':
                    # PyTorch SVD - ensure operations happen on GPU
                    U, S, Vt = torch.linalg.svd(gpu_matrix, full_matrices=False)
                    # Synchronize GPU operations
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    elif self.device.type == 'mps':
                        torch.mps.synchronize()
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                # Truncate to n_components
                U = U[:, :n_components]
                S = S[:n_components]
                Vt = Vt[:n_components, :]
                
                # Convert back to CPU
                return self.to_cpu(U), self.to_cpu(S), self.to_cpu(Vt)
            except Exception as e:
                warnings.warn(f"GPU SVD failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        # CPU fallback using sklearn
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=n_components, random_state=random_state, n_iter=5)
        U = svd.fit_transform(matrix)
        S = svd.singular_values_
        Vt = svd.components_
        return U, S, Vt
    
    def pca(self, matrix: np.ndarray, n_components: int,
            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform PCA on GPU or CPU.
        
        Args:
            matrix: Input matrix (n_samples, n_features)
            n_components: Number of components
            random_state: Random seed
            
        Returns:
            (transformed, components, mean)
        """
        if self.use_gpu:
            try:
                gpu_matrix = self.to_gpu(matrix)
                
                # Verify data is on GPU
                if self.backend == 'cupy':
                    if not isinstance(gpu_matrix, cp.ndarray):
                        raise RuntimeError("Matrix not on GPU (CuPy)")
                elif self.backend == 'torch':
                    if not isinstance(gpu_matrix, torch.Tensor) or gpu_matrix.device.type != self.device.type:
                        raise RuntimeError(f"Matrix not on GPU (PyTorch): device={gpu_matrix.device if isinstance(gpu_matrix, torch.Tensor) else 'not tensor'}")
                
                # Center the data
                if self.backend == 'cupy':
                    mean = cp.mean(gpu_matrix, axis=0)
                    centered = gpu_matrix - mean
                    # SVD
                    U, S, Vt = cp.linalg.svd(centered, full_matrices=False)
                    # Transform: U @ diag(S)
                    transformed = U @ cp.diag(S)
                elif self.backend == 'torch':
                    mean = torch.mean(gpu_matrix, axis=0)
                    centered = gpu_matrix - mean
                    # SVD
                    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
                    # Transform: U @ diag(S)
                    transformed = U @ torch.diag(S)
                    # Synchronize GPU operations
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    elif self.device.type == 'mps':
                        torch.mps.synchronize()
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                # Truncate
                U = U[:, :n_components]
                S = S[:n_components]
                Vt = Vt[:n_components, :]
                transformed = transformed[:, :n_components]
                
                # Convert back to CPU
                return self.to_cpu(transformed), self.to_cpu(Vt), self.to_cpu(mean)
            except Exception as e:
                warnings.warn(f"GPU PCA failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        # CPU fallback using sklearn
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        transformed = pca.fit_transform(matrix)
        return transformed, pca.components_, pca.mean_
    
    def matmul(self, a, b):
        """Matrix multiplication on GPU or CPU."""
        if self.use_gpu:
            try:
                a_gpu = self.to_gpu(a) if isinstance(a, np.ndarray) else a
                b_gpu = self.to_gpu(b) if isinstance(b, np.ndarray) else b
                
                # Verify inputs are on GPU
                if self.backend == 'torch':
                    if isinstance(a_gpu, torch.Tensor) and a_gpu.device.type != self.device.type:
                        raise RuntimeError(f"Tensor a not on GPU: {a_gpu.device}")
                    if isinstance(b_gpu, torch.Tensor) and b_gpu.device.type != self.device.type:
                        raise RuntimeError(f"Tensor b not on GPU: {b_gpu.device}")
                
                if self.backend == 'cupy':
                    result = cp.matmul(a_gpu, b_gpu)
                elif self.backend == 'torch':
                    result = torch.matmul(a_gpu, b_gpu)
                    # Synchronize GPU operations
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    elif self.device.type == 'mps':
                        torch.mps.synchronize()
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                return self.to_cpu(result)
            except Exception as e:
                warnings.warn(f"GPU matmul failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        return np.matmul(a, b)
    
    def sparse_matmul(self, a, b):
        """Sparse matrix multiplication on GPU or CPU."""
        if self.use_gpu:
            try:
                # PyTorch sparse support is limited, so prefer CuPy for sparse ops
                if self.backend == 'cupy':
                    a_gpu = self.sparse_to_gpu(a) if isinstance(a, sparse.spmatrix) else a
                    b_gpu = self.to_gpu(b) if isinstance(b, np.ndarray) else b
                    result = a_gpu @ b_gpu
                    return self.to_cpu(result)
                elif self.backend == 'torch':
                    # PyTorch: convert sparse to dense for now (MPS doesn't support sparse well)
                    if isinstance(a, sparse.spmatrix):
                        a_dense = self.to_gpu(a.toarray())
                        b_gpu = self.to_gpu(b) if isinstance(b, np.ndarray) else b
                        result = torch.matmul(a_dense, b_gpu)
                        return self.to_cpu(result)
            except Exception as e:
                warnings.warn(f"GPU sparse matmul failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        if isinstance(a, sparse.spmatrix):
            return a @ b
        return np.matmul(a, b)
    
    def norm(self, arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        """Compute L2 norm on GPU or CPU."""
        if self.use_gpu:
            try:
                gpu_arr = self.to_gpu(arr)
                
                if self.backend == 'cupy':
                    result = cp.linalg.norm(gpu_arr, axis=axis, keepdims=keepdims)
                elif self.backend == 'torch':
                    result = torch.linalg.norm(gpu_arr, dim=axis, keepdim=keepdims)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                return self.to_cpu(result)
            except Exception as e:
                warnings.warn(f"GPU norm failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        return np.linalg.norm(arr, axis=axis, keepdims=keepdims)
    
    def sum(self, arr, axis: Optional[int] = None, keepdims: bool = False):
        """Sum array on GPU or CPU."""
        # Handle sparse matrices - convert to dense or use CPU
        if sparse.issparse(arr):
            # For sparse matrices, use CPU (more efficient for sparse ops)
            return np.array(arr.sum(axis=axis)).flatten() if axis is not None else arr.sum()
        
        if self.use_gpu:
            try:
                # Only convert numpy arrays to GPU
                if isinstance(arr, np.ndarray):
                    gpu_arr = self.to_gpu(arr)
                else:
                    # If it's already a GPU array/tensor, use it directly
                    gpu_arr = arr
                
                if self.backend == 'cupy':
                    result = cp.sum(gpu_arr, axis=axis, keepdims=keepdims)
                elif self.backend == 'torch':
                    # PyTorch sum: dim can be int or tuple, keepdim (not keepdims)
                    if axis is not None:
                        # Convert axis to tuple if single int for consistency
                        dim = axis if isinstance(axis, (tuple, list)) else (axis,)
                        result = torch.sum(gpu_arr, dim=dim, keepdim=keepdims)
                    else:
                        result = torch.sum(gpu_arr)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                return self.to_cpu(result) if hasattr(result, 'cpu') or isinstance(result, cp.ndarray) else result
            except Exception as e:
                warnings.warn(f"GPU sum failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        return np.sum(arr, axis=axis, keepdims=keepdims)
    
    def maximum(self, a, b):
        """Element-wise maximum on GPU or CPU."""
        if self.use_gpu:
            try:
                # Convert inputs to GPU tensors/arrays
                if isinstance(a, np.ndarray):
                    a_gpu = self.to_gpu(a)
                elif isinstance(a, (int, float)) and self.backend == 'torch':
                    # Convert scalar to tensor on same device as 'a' if 'a' is already a tensor
                    # Otherwise create on default device
                    a_gpu = torch.tensor(a, dtype=torch.float32, device=self.device)
                else:
                    a_gpu = a
                
                if isinstance(b, np.ndarray):
                    b_gpu = self.to_gpu(b)
                elif isinstance(b, (int, float)):
                    # Convert scalar to tensor/array
                    if self.backend == 'torch':
                        # If a_gpu is a tensor, use its device and dtype
                        if isinstance(a_gpu, torch.Tensor):
                            b_gpu = torch.tensor(b, dtype=a_gpu.dtype, device=a_gpu.device)
                        else:
                            b_gpu = torch.tensor(b, dtype=torch.float32, device=self.device)
                    elif self.backend == 'cupy':
                        b_gpu = cp.array(b)
                    else:
                        b_gpu = b
                else:
                    b_gpu = b
                
                if self.backend == 'cupy':
                    result = cp.maximum(a_gpu, b_gpu)
                elif self.backend == 'torch':
                    result = torch.maximum(a_gpu, b_gpu)
                else:
                    raise ValueError(f"Unknown backend: {self.backend}")
                
                return self.to_cpu(result)
            except Exception as e:
                warnings.warn(f"GPU maximum failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        return np.maximum(a, b)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup GPU memory."""
        if self.use_gpu:
            try:
                if self.backend == 'cupy':
                    cp.get_default_memory_pool().free_all_blocks()
                elif self.backend == 'torch':
                    torch.mps.empty_cache() if self.device_type == 'mps' else torch.cuda.empty_cache()
            except:
                pass
        return False

