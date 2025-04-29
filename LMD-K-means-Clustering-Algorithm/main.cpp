//
//  main.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/10/25.
//

#include <iostream>
#include <fstream>
#include <vector>

#include "kmeans.hpp"
#include "timer.hpp"
#include "utility.hpp"


// g++-14 main.cpp -fopenmp -o main

int main() {
  std::vector<std::vector<float>> images;
  std::vector<int> labels;
  
  {
    Timer timer("load_MNIST");
    load_MNIST("/Users/palmi/XcodeProjects/LMD-K-means-Clustering-Algorithm/LMD-K-means-Clustering-Algorithm/data/mnist-images.txt", "/Users/palmi/XcodeProjects/LMD-K-means-Clustering-Algorithm/LMD-K-means-Clustering-Algorithm/data/mnist-labels.txt", images, labels);
  }
 
  KMeans kmeans(images, labels, 10, 784);
  kmeans.test();
  
  
  /*std::cout << "No. Images: " << images.size() << std::endl;
  for (int i=0; i<28; i++) {
    for (int j=0; j<28; j++)
      std::cout<<images[0][i*28+j] << " ";
    std::cout << std::endl;
  }
  
  std::cout << "Image is " << labels[0] << std::endl;*/
  
  return 0;
}
