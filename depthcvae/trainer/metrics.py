import numpy as np

def RMSE(input: np.ndarray, target: np.ndarray):
    """
    Computes the root mean squared error between input and target
    Args:
        input: (HxW) or (NxHxW) numpy array
        target: (HxW) or (NxHxW) numpy array 
    """
    return np.sqrt(np.mean((input-target)**2))

def logRMSE(input: np.ndarray, target: np.ndarray, eps=1e-6):
    """
    Computes the root mean squared error between log(input) and log(target)
    Args:
        input: (HxW) or (NxHxW) numpy array
        target: (HxW) or (NxHxW) numpy array  
    """
    input_clipped = np.clip(np.copy(input), a_min=eps, a_max=None)
    target_clipped = np.clip(np.copy(target), a_min=eps, a_max=None)

    return np.sqrt(np.mean(np.log(input_clipped)-np.log(target_clipped))**2)

def ARE(input: np.ndarray, target: np.ndarray, eps=1e-6):
    """
    Computes the absolute relative error between input and target
    Args:
        input: (HxW) or (NxHxW) numpy array
        target: (HxW) or (NxHxW) numpy array 
    """
    input_clipped = np.clip(np.copy(input), a_min=eps, a_max=None)
    target_clipped = np.clip(np.copy(target), a_min=eps, a_max=None)

    return np.mean(np.abs(input_clipped - target_clipped) / target_clipped)

def SRE(input: np.ndarray, target: np.ndarray, eps=1e-6):
    """
    Computes the squared relative error between input and target
    Args:
        input: (HxW) or (NxHxW) numpy array
        target: (HxW) or (NxHxW) numpy array 
    """
    input_clipped = np.clip(np.copy(input), a_min=eps, a_max=None)
    target_clipped = np.clip(np.copy(target), a_min=eps, a_max=None)

    return np.mean((input_clipped - target_clipped)**2 / target_clipped)

def accuracy(input: np.ndarray, target: np.ndarray, threshold=1.25, eps=1e-6):
    """
    Computes the accuracy: % of input such that max(input/target, target/input) < threshold
    Args:
        input: (HxW) or (NxHxW) numpy array
        target: (HxW) or (NxHxW) numpy array 
    """
    input_clipped = np.clip(np.copy(input), a_min=eps, a_max=None)
    target_clipped = np.clip(np.copy(target), a_min=eps, a_max=None)

    delta = np.maximum(input_clipped / target_clipped, target_clipped / input_clipped)

    return np.mean(delta < threshold)