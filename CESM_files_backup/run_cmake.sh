cd bld
rm -rf *
CC=mpicc CXX=mpicxx FC=mpif90 cmake -DCMAKE_INSTALL_PREFIX=~/dev/install_pr/ParallelIO/ -DZ5_DIR=~/miniconda3/envs/z5-py36/include -DWITH_ZLIB=ON -DZLIB_ROOT=/glade/u/home/weile/miniconda3/envs/z5-py36/ -DBOOST_ROOT=~/miniconda3/envs/z5-py36 -DWITH_MARRAY=ON -DMARRAY_INCLUDE_DIR=~/miniconda3/envs/z5-py36/include -DCMAKE_VERBOSE_MAKEFILE=ON ..
make V=1
#make test_z5_create_file V=1 >& tmp
