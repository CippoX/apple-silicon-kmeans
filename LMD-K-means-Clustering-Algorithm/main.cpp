//
//  main.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/10/25.
//

#include <iostream>
#include <fstream>
#include <vector>

#include "utility.hpp"
#include "timer.hpp"
#include "kmeans.hpp"

int main() {
  std::vector< std::vector<float> > images;
  std::vector<int> labels;
  
  KMeans kmeans;
  
  {
    Timer timer("load_MNIST");
    load_MNIST("/Users/palmi/XcodeProjects/LMD-K-means-Clustering-Algorithm/LMD-K-means-Clustering-Algorithm/data/mnist-images.txt", "/Users/palmi/XcodeProjects/LMD-K-means-Clustering-Algorithm/LMD-K-means-Clustering-Algorithm/data/mnist-labels.txt", images, labels);
  }
  
  {
    Timer timer("Euclidean distance test");
    std::cout << kmeans.euclideanDistance(images[0], images[1]) << std::endl;
  }
  
  {
    Timer timer("Optimized euclidean distance test");
    std::cout << kmeans.optimizedEuclideanDistance(images[0], images[1]) << std::endl;
  }
  
  /*std::cout << "No. Images: " << images.size() << std::endl;
  for (int i=0; i<28; i++) {
    for (int j=0; j<28; j++)
      std::cout<<images[0][i*28+j] << " ";
    std::cout << std::endl;
  }
  
  std::cout << "Image is " << labels[0] << std::endl;*/
  
  return 0;
}
