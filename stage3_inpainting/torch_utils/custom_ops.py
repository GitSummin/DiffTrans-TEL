# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob
import torch
import torch.utils.cpp_extension
import importlib
import hashlib
import shutil
from pathlib import Path

from torch.utils.file_baton import FileBaton

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief' # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']

    # Already cached?
    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    # Print status.
    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)

    try:
        # Ensure CUDA architecture is set
        if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
            os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Set architecture for RTX 30 series

        # Make sure we can find the necessary compiler binaries.
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] = compiler_bindir + ';' + os.environ['PATH']

        # Add extra CUDA flags if not already present
        if 'extra_cuda_cflags' not in build_kwargs:
            build_kwargs['extra_cuda_cflags'] = ['--use-fast-math', '-allow-unsupported-compiler']

        # Compile and load.
        verbose_build = (verbosity == 'full')
        torch.utils.cpp_extension.load(
            name=module_name,
            verbose=verbose_build,
            sources=sources,
            **build_kwargs
        )
        module = importlib.import_module(module_name)

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    # Print status and add to cache.
    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module

#----------------------------------------------------------------------------
