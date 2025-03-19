//
//  kmeans.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/18/25.
//

#include "kmeans.hpp"


class KMeansException : public std::exception {
private:
  std::string message;
  
public:
  KMeansException(const char* msg) : message(msg){
  }
  const char* what() const throw() {
    return message.c_str();
  }
};



float KMeans::euclideanDistance(std::vector<float> v1, std::vector<float> v2) {
  if (v1.size() != v2.size()){
    throw KMeansException("The two vectors must have the same dimension.");
  }
  
  float sum = 0.0;
  
  for(int i = 0; i < v1.size(); i++) {
    sum += std::pow(v1[i] - v2[i], 2);
  }
  
  return std::sqrt(sum);
}


/**

 */
float KMeans::optimizedEuclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
  float sum = 0.0f;
  size_t size = v1.size();
  size_t i = 0;
  
  if (size != v2.size()){
    throw KMeansException("The two vectors must have the same dimension.");
  }
  
  for(; i + 3 < size; i += 4) {
    float32x4_t simd_v1 = vld1q_f32(&v1[i]);
    float32x4_t simd_v2 = vld1q_f32(&v2[i]);
    float32x4_t simd_diff = vsubq_f32(simd_v1, simd_v2);
    float32x4_t simd_squared = vmulq_f32(simd_diff, simd_diff);
    
    float *res = (float *)&simd_squared;
    
    sum += res[0] + res[1] + res[2] + res[3];
  }
  
  for (; i < size; i++) {
      float diff = v1[i] - v2[i];
      sum += diff * diff;
  }
  
  return std::sqrt(sum);
}



std::vector<float> KMeans::calculateCentroid(const std::vector< std::vector<float> >& vectors) {
  std::vector<float> mean( vectors[0].size(), 0);
  
  for(size_t i = 0; i < vectors.size(); i++) {
    for(size_t j = 0; j < vectors[i].size(); j++) {
      mean[j] += vectors[i][j];
    }
  }
  
  for(size_t i = 0; i < mean.size(); i++) {
    mean[i] /= vectors.size();
  }
  
  return mean;
}



std::vector<float> KMeans::optimizedCalculateCentroid(const std::vector< std::vector<float> >& vectors) {
  size_t number_of_vectors = vectors.size();
  size_t vectors_space_dimension = vectors[0].size();
  std::vector<float> mean( vectors_space_dimension, 0);
  
  for(size_t i = 0; i < number_of_vectors; i++) {
    size_t j = 0;
    for(; j + 3 < vectors_space_dimension; j += 4) {
      float32x4_t simd_v = vld1q_f32(&vectors[i][j]);
      float32x4_t simd_mean = vld1q_f32(&mean[j]);
      
      float32x4_t simd_sum = vaddq_f32(simd_v, simd_mean);
      
      vst1q_f32(&mean[j], simd_sum);
    }
    
    for (; j < vectors_space_dimension; j++) {
      mean[j] += vectors[i][j];
    }
  }
  
  size_t i = 0;
  float32x4_t inv_num_vectors = vdupq_n_f32(1.0f / number_of_vectors);
  
  for (; i + 4 <= vectors_space_dimension; i += 4) {
    float32x4_t simd_mean = vld1q_f32(&mean[i]);
    float32x4_t simd_div = vmulq_f32(simd_mean, inv_num_vectors);
    vst1q_f32(&mean[i], simd_div);
  }

  for (; i < vectors_space_dimension; i++) {
    mean[i] *= (1.0f / number_of_vectors);
  }
  
  return mean;
}
