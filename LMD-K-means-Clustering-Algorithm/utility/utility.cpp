//
//  utility.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/17/25.
//

#include <iostream>
#include <thread>
#include <chrono>

#include "utility.hpp"

// g++-14 -std=c++11 -O3 -fopenmp kmeans.cpp -o kmeans

void load_MNIST(    const char* images_file, const char* labels_file,
                std::vector< std::vector<float> > &images,
                std::vector<int> &labels ) {
  int rows = 70000, cols=784;
  
  std::ifstream file(images_file);
  if (!file) {
    std::cerr << "Error opening file!" << std::endl;
    return;
  }
  
  // resize matrix
  images.resize(rows);
  for (auto &i : images)
    i.resize(cols);
  
  // Read the matrix elements
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      file >> images[i][j];
    }
  }
  
  file.close();
  
  std::ifstream file2(labels_file);
  if (!file2) {
    std::cerr << "Error opening file!" << std::endl;
    return;
  }
  
  // resize matrix
  labels.resize(rows);
  
  // Read the matrix elements
  for (int i = 0; i < rows; i++)
    file2 >> labels[i];
  
  file2.close();
}



template <typename T>

std::set<T> setFromVector(std::vector<T> v) {
  std::set<std::string> set(v.begin(), v.end());
  return set;
}
