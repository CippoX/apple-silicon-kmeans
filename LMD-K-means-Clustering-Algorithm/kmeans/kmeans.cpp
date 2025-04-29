//
//  kmeans.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/18/25.
//

#include "kmeans.hpp"
#include "timer.hpp"
#include <random>


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



KMeans::KMeans(const std::vector< std::vector<float>> &_images,
               const std::vector<int> &_labels,
               size_t _number_of_centroids,
               size_t _vectorspace_dimension) {
  
  images = _images;
  labels = _labels;
  number_of_centroids = _number_of_centroids;
  vectorspace_dimension = _vectorspace_dimension;
}



float KMeans::euclideanDistance(std::vector<float> v1, std::vector<float> v2) {
  if (v1.size() != v2.size()){
    throw KMeansException("The two vectors must have the same dimension.");
  }
  
  float sum = 0.0;
  
  for(int i = 0; i < vectorspace_dimension; i++) {
    sum += std::pow(v1[i] - v2[i], 2);
  }
  
  return std::sqrt(sum);
}


/**

 */
float KMeans::optimizedEuclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
  float sum = 0.0f;
  size_t i = 0;

  for(; i + 7 < vectorspace_dimension; i += 8) {
    float32x4x2_t simd_v1 = vld1q_f32_x2(&v1[i]);
    float32x4x2_t simd_v2 = vld1q_f32_x2(&v2[i]);
    
    float32x4x2_t simd_diff;
    simd_diff.val[0] = vsubq_f32(simd_v1.val[0], simd_v2.val[0]);
    simd_diff.val[1] = vsubq_f32(simd_v1.val[1], simd_v2.val[1]);
    
    float32x4x2_t simd_squared;
    simd_squared.val[0] = vmulq_f32(simd_diff.val[0], simd_diff.val[0]);
    simd_squared.val[1] = vmulq_f32(simd_diff.val[1], simd_diff.val[1]);
    
    float *res = (float *)&simd_squared;
    
    sum += res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[3];
  }
  
  for (; i < vectorspace_dimension; i++) {
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



std::vector<float> KMeans::optimizedCalculateCentroid(const std::vector<std::vector<float>>& vectors) {
  size_t number_of_vectors = vectors.size();
  std::vector<float> mean(vectorspace_dimension, 0);

  for (size_t i = 0; i < number_of_vectors; i++) {
    size_t j = 0;
    
    for (; j + 7 < vectorspace_dimension; j += 8) {
      float32x4x2_t simd_v = vld1q_f32_x2(&vectors[i][j]);
      float32x4x2_t simd_mean = vld1q_f32_x2(&mean[j]);
      simd_mean.val[0] = vaddq_f32(simd_mean.val[0], simd_v.val[0]);
      simd_mean.val[1] = vaddq_f32(simd_mean.val[1], simd_v.val[1]);
      
      vst1q_f32_x2(&mean[j], simd_mean);
    }

    for (; j < vectorspace_dimension; j++) {
      mean[j] += vectors[i][j];
    }
  }

  size_t i = 0;
  float32x4_t inv_num_vectors = vdupq_n_f32(1.0f / number_of_vectors);
  
  for (; i + 7 < vectorspace_dimension; i += 8) {
    float32x4x2_t simd_mean = vld1q_f32_x2(&mean[i]);
    simd_mean.val[0] = vmulq_f32(simd_mean.val[0], inv_num_vectors);
    simd_mean.val[1] = vmulq_f32(simd_mean.val[1], inv_num_vectors);

    vst1q_f32_x2(&mean[i], simd_mean);
  }

  for (; i < vectorspace_dimension; i++) {
    mean[i] *= (1.0f / number_of_vectors);
  }
  
  return mean;
}






void KMeans::test() {  
  
  {
    Timer timer("Euclidean distance test");
    euclideanDistance(images[0], images[1]);
  }
  
  {
    Timer timer("Optimized euclidean distance test");
    optimizedEuclideanDistance(images[0], images[1]);
  }
  
  
  {
    Timer timer("Centroid calculation test");
    std::vector<float> v = calculateCentroid(images);
    
    float sum = 0.0f;
    for (float e : v) {
      sum += e;
    }
    std::cout << "sum of means " << sum << std::endl;
  }
  
  {
    Timer timer("Optimized centroid calculation test");
    std::vector<float> v = optimizedCalculateCentroid(images);
    
    float sum = 0.0f;
    for (float e : v) {
      sum += e;
    }
    std::cout << "sum of means " << sum << std::endl;
  }
  
  kmeans_pp();
  
  for (int k=0; k<centroids.size(); k++) {
    for (int i=0; i<28; i++) {
      for (int j=0; j<28; j++)
        std::cout<<centroids[k][i*28+j] << " ";
      std::cout << std::endl;
    }
  }
}



float KMeans::distanceFromClosestCentroid(const std::vector<float> &point) {
  float minimum_distance = std::numeric_limits<float>::max();
  
  for(size_t i = 0; i < centroids.size(); i++) {
    minimum_distance = std::min(minimum_distance, optimizedEuclideanDistance(point, centroids[i]));
  }
  
  return minimum_distance;
}



// Initialize the centroids using k-means++
void KMeans::kmeans_pp() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> uni(0, images.size() - 1);
  
  // Choose the first centroid uniformly from the data points
  size_t first = uni(gen);
  centroids.push_back(images[first]);
  std::cout << first << std::endl;
  
  // Select all the other centroids
  for (size_t c = 0; c < number_of_centroids - 1; c++) {
    std::vector<float> squared_distances;

    for (size_t i = 0; i < images.size(); i++) {
      float minimum_distance = distanceFromClosestCentroid(images[i]);
      squared_distances.push_back(minimum_distance * minimum_distance);
    }

    std::discrete_distribution<size_t> weighted_dist(squared_distances.begin(), squared_distances.end());
    size_t next_centroid_index = weighted_dist(gen);
    
    std::cout << next_centroid_index << std::endl;
    
    centroids.push_back(images[next_centroid_index]);
  }
}
