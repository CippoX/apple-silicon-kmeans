# Apple Silicon K-Means
This repo contains a C++ implementation of Mini-Batch K-Means optimized for Apple Silicon Processors.

# Compilation Instructions
 
 ### Standard Compilation:

  g++ main.cpp kmeans/kmeans.cpp kmeans/mini-batch-kmeans.cpp \
      kmeans/parallel-mini-batch-kmeans.cpp timer/timer.cpp \
      utility/utility.cpp -fopenmp -o main
 
 
###  Apple Silicon (macOS) Compilation:

  Note: ARM NEON does come by default on macOS for Apple Silicon.
 
  1. Install LLVM via Homebrew:\
     brew install llvm\
     (Make sure to use the Apple Silicon version of Homebrew)
 
  2. Set up environment variables:\
     export CC=/opt/homebrew/opt/llvm/bin/clang\
     export CXX=/opt/homebrew/opt/llvm/bin/clang++\
     export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"\
     export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"\
     export PATH="/opt/homebrew/opt/llvm/bin:$PATH"\
 
  3. Compile with clang++:\
     clang++ main.cpp kmeans/kmeans.cpp kmeans/mini-batch-kmeans.cpp \
             kmeans/parallel-mini-batch-kmeans.cpp timer/timer.cpp \
             utility/utility.cpp -fopenmp -o main
 
  4, You  may specify the number of threads at runtime with
     OMP_NUM_THREADS=4 ./main num_clusters batch_size
