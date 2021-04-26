export DGLBACKEND=$1
export DGLTESTDEV=$2
export DGL_LIBRARY_PATH=${PWD}/build
export PYTHONPATH=tests:${PWD}/python:$PYTHONPATH
export DGL_DOWNLOAD_DIR=${PWD}
export TF_FORCE_GPU_ALLOW_GROWTH=true

if [ $2 == "gpu" ] 
then
  export CUDA_VISIBLE_DEVICES=0
else
  export CUDA_VISIBLE_DEVICES=-1
fi


python3 -m pytest -v -s --junitxml=pytest_compute.xml tests/compute/test_update_all_hetero.py 
#python3 -m pytest -v --junitxml=pytest_backend.xml tests/$DGLBACKEND || fail "backend-specific"

#export OMP_NUM_THREADS=1
#if [ $2 != "gpu" ]; then
#    python3 -m pytest -v --junitxml=pytest_distributed.xml tests/distributed || fail "distributed"
#fi
