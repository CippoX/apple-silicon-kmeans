//
//  utility.hpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/10/25.
//

#ifndef utility_hpp
#define utility_hpp

#include <iostream>
#include <fstream>
#include <vector>
#include <set>
#include <thread>
#include <chrono>
#include <arm_neon.h>
#include <random>
#include <algorithm>
#include <cmath>
#include <utility>

void load_MNIST(    const char* images_file, const char* labels_file,
                std::vector< std::vector<float> > &images,
                std::vector<int> &labels );

std::vector<float> operator*(float scalar, const std::vector<float>& v);
std::vector<float> operator+(const std::vector<float>& vec1, const std::vector<float>& vec2);

template <typename T>
std::set<T> setFromVector(const std::vector<T>& v) {
  std::set<T> set(v.begin(), v.end());
  return set;
}

template <typename T, typename U>
std::pair<std::vector<std::vector<T>>, std::vector<U>> expandDataset(const std::vector<std::vector<T>>& dataset, const std::vector<U>& labels, size_t to) {
  if (dataset.empty() || to <= dataset.size()) {
    return {dataset, labels};
  }
  
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> uni(0, 99);
  
  std::vector<std::vector<T>> expanded_dataset = dataset;
  std::vector<U> expanded_labels = labels;
  
  size_t vectorspace_dimension = dataset[0].size();
  size_t original_size = dataset.size();
  
  for (size_t i = 0; i < to - original_size; i++) {
    std::vector<T> aux = dataset[i % original_size];
    U aux_label = labels[i % labels.size()];
    
    for (size_t j = 0; j < vectorspace_dimension; j++) {
      size_t prob = uni(gen);
      
      if (prob >= 90) {
        aux[j] = static_cast<T>(abs(static_cast<T>(aux[j]) - 255));
      }
    }
    
    expanded_dataset.push_back(aux);
    expanded_labels.push_back(aux_label);
  }
  
  return {expanded_dataset, expanded_labels};
}

#endif /* utility_hpp */
