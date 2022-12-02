
#CUDA_LIB_PATH=/opt/hpc_software/sdk/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64/stubs CC=gcc CXX=g++ python3 buildbot/configure.py --cuda --cmake-opt="-DCUDA_TOOLKIT_ROOT_DIR=/opt/hpc_software/sdk/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7" --cmake-gen "Unix Makefiles" -o build_cuda -n 72
# CUDA_LIB_PATH=/opt/hpc_software/sdk/nvidia/hpc_sdk/Linux_x86_64/22.7/cuda/11.7/lib64/stubs CC=gcc CXX=g++ python3 buildbot/compile.py -j 72 -o build_cuda


LLVM_CUDA=/nfs/site/home/phuongn2/sycl_workspace/llvm/build_cuda
export PATH=$LLVM_CUDA/bin:$PATH
export CPATH=$LLVM_CUDA/include:$CPATH
export LD_LIBRARY_PATH=$LLVM_CUDA/lib:$LD_LIBRARY_PATH
export CPATH=$LLVM_CUDA/include/sycl:$CPATH

# FLAGS for A100 - -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_60

