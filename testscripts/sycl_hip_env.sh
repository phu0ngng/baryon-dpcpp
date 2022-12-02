
# Compile LLVM
# This checkout worked 99ee13b04018a9271ff07ef3d7a416c7665e0ee3
#CC=gcc CXX=g++ python3 buildbot/configure.py --hip --hip-platform AMD -o build_hip -t RELEASE --cmake-gen "Unix Makefiles" --cmake-opt=-DSYCL_BUILD_PI_HIP_ROCM_DIR=/opt/hpc_software/sdk/amd/rocm-4.5.2
#python3 buildbot/compile.py -j 72 -o build_hip

LLVM_HIP=/nfs/site/home/phuongn2/sycl_workspace/llvm/build_hip
export PATH=$LLVM_HIP/bin:$PATH
export CPATH=$LLVM_HIP/include:$CPATH
export LD_LIBRARY_PATH=$LLVM_HIP/lib:$LD_LIBRARY_PATH
export CPATH=$LLVM_HIP/include/sycl:$CPATH

# FLAGS for MI210 and MI250 -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a
