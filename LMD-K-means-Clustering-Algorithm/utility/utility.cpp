//
//  utility.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/17/25.
//

#include <iostream>
#include <thread>
#include <chrono>
#include <arm_neon.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>


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



std::vector<float> operator*(float scalar, const std::vector<float>& v) {
  std::vector<float> result(v.size());
  
  size_t i = 0;
  float32x4_t scalar_vec = vdupq_n_f32(scalar);
  
  // SIMD processing - 4 elements at a time
  for (; i + 3 < v.size(); i += 4) {
    float32x4_t v_chunk = vld1q_f32(&v[i]);
    float32x4_t result_chunk = vmulq_f32(scalar_vec, v_chunk);
    vst1q_f32(&result[i], result_chunk);
  }
  
  // Handle remaining elements
  for (; i < v.size(); i++) {
    result[i] = scalar * v[i];
  }
  
  return result;
}



std::vector<float> operator+(const std::vector<float>& v1, const std::vector<float>& v2) {
  if (v1.size() != v2.size()) {
    throw std::invalid_argument("Vector sizes must match for addition");
  }
  
  std::vector<float> result(v1.size());
  
  size_t i = 0;
  
  // SIMD processing - 4 elements at a time
  for (; i + 3 < v1.size(); i += 4) {
    float32x4_t v1_chunk = vld1q_f32(&v1[i]);
    float32x4_t v2_chunk = vld1q_f32(&v2[i]);
    float32x4_t result_chunk = vaddq_f32(v1_chunk, v2_chunk);
    vst1q_f32(&result[i], result_chunk);
  }
  
  // Handle remaining elements
  for (; i < v1.size(); i++) {
    result[i] = v1[i] + v2[i];
  }
  
  return result;
}



std::vector<std::vector<float>> expandDataset(std::vector<std::vector<float>> const& dataset, size_t to) {
  if (dataset.empty() || to <= dataset.size()) {
    return dataset;
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> uni(0, 99);
  
  std::vector<std::vector<float>> expanded_dataset = dataset;
  size_t vectorspace_dimension = dataset[0].size();

  for (size_t i = 0; i < to - dataset.size(); i++) {
    std::vector<float> aux = dataset[i % dataset.size()];
    
    for(size_t j = 0; j < vectorspace_dimension; j++) {
      size_t prob = uni(gen);
      
      if(prob >= 80) {
        aux[j] = 255;
      } else if(prob <= 19) {
        aux[j] = 0;
      }
    }
    
    expanded_dataset.push_back(aux);
  }
  
  return expanded_dataset;
}
