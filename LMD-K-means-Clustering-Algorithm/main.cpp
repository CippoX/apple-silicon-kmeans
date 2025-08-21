//
//  main.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/10/25.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <omp.h>

#include "kmeans/kmeans.hpp"
#include "timer/timer.hpp"
#include "utility/utility.hpp"
#include "kmeans/mini-batch-kmeans.hpp"
#include "kmeans/parallel-mini-batch-kmeans.hpp"

/**
  COMPILATION INSTRUCTIONS
  ========================
 
  Standard Compilation:
  ------------------------------------
  g++ main.cpp kmeans/kmeans.cpp kmeans/mini-batch-kmeans.cpp \
      kmeans/parallel-mini-batch-kmeans.cpp timer/timer.cpp \
      utility/utility.cpp -fopenmp -o main
 
 
  Apple Silicon (macOS) Compilation:
  ---------------------------------
  Note: ARM NEON does come by default on macOS for Apple Silicon.
 
  1. Install LLVM via Homebrew:
     brew install llvm
     (Make sure to use the Apple Silicon version of Homebrew)
 
  2. Set up environment variables:
     export CC=/opt/homebrew/opt/llvm/bin/clang
     export CXX=/opt/homebrew/opt/llvm/bin/clang++
     export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
     export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"
     export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
 
  3. Compile with clang++:
     clang++ main.cpp kmeans/kmeans.cpp kmeans/mini-batch-kmeans.cpp \
             kmeans/parallel-mini-batch-kmeans.cpp timer/timer.cpp \
             utility/utility.cpp -fopenmp -o main
 
  4, You  may specify the number of threads at runtime with
     OMP_NUM_THREADS=4 ./main
 */



int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <num_clusters> <batch_size>\n";
    return 1;
  }
  
  int num_clusters = std::atoi(argv[1]);
  int batch_size = std::atoi(argv[2]);
  
  std::vector<std::vector<float>> images;
  std::vector<int> labels;
  
  load_MNIST("/Users/palmi/XcodeProjects/LMD-K-means-Clustering-Algorithm/LMD-K-means-Clustering-Algorithm/data/mnist-images.txt", "/Users/palmi/XcodeProjects/LMD-K-means-Clustering-Algorithm/LMD-K-means-Clustering-Algorithm/data/mnist-labels.txt", images, labels);
  
  images = expandDataset(images, 200000);
  
  int num_features = 784;
  
  ParallelMiniBatchKMeans kmeans(images, labels, num_clusters, num_features, batch_size);
  
  {
    Timer timer("Total Clustering Time");
    kmeans.test();
  }
  
  return 0;
}
