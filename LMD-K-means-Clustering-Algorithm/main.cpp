//
//  main.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/10/25.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <omp.h>

#include "kmeans/kmeans.hpp"
#include "timer/timer.hpp"
#include "utility/utility.hpp"
#include "kmeans/mini-batch-kmeans.hpp"
#include "kmeans/parallel-mini-batch-kmeans.hpp"

/// Compile with
/**
 g++ main.cpp kmeans/kmeans.cpp kmeans/mini-batch-kmeans.cpp kmeans/parallel-mini-batch-kmeans.cpp timer/timer.cpp utility/utility.cpp -fopenmp -o main
 
 or
 
 clang++ main.cpp kmeans/kmeans.cpp kmeans/mini-batch-kmeans.cpp kmeans/parallel-mini-batch-kmeans.cpp timer/timer.cpp utility/utility.cpp -fopenmp -o main
 */

int main() {
  std::vector<std::vector<float>> images;
  std::vector<int> labels;
  
  {
    Timer timer("load_MNIST");
    load_MNIST("/Users/palmi/XcodeProjects/LMD-K-means-Clustering-Algorithm/LMD-K-means-Clustering-Algorithm/data/mnist-images.txt", "/Users/palmi/XcodeProjects/LMD-K-means-Clustering-Algorithm/LMD-K-means-Clustering-Algorithm/data/mnist-labels.txt", images, labels);
  }
  
  ParallelMiniBatchKMeans kmeans(images, labels, 10, 784, 70000);
  kmeans.test();
  
  return 0;
}
