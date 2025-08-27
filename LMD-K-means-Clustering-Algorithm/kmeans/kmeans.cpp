//
//  kmeans.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/18/25.
//

#include "kmeans.hpp"
#include "../timer/timer.hpp"
#include <random>
#include <thread>
#include <omp.h>

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
  clusters = std::vector<size_t>(_images.size(), -1);
}



/**===----------------------------------------------------------------------===

 - Utility Functions
 - euclideanDistance
 - optimizedEuclideanDistance
 - calculateCentroid
 - optimizedCalculateCentroid
 - distanceFromClosestCentroid
 
===----------------------------------------------------------------------=== **/

float KMeans::euclideanDistance(std::vector<float> v1, std::vector<float> v2) {
  if (v1.size() != v2.size()){
    throw KMeansException("The two vectors must have the same dimension.");
  }
  
  float sum = 0.0f;
  
  for(size_t i = 0; i < vectorspace_dimension; i++)
    sum += std::pow(v1[i] - v2[i], 2);
  
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
    
    sum += res[0] + res[1] + res[2] + res[3] + res[4] + res[5] + res[6] + res[7];
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
  
  for(size_t i = 0; i < mean.size(); i++)
    mean[i] /= vectors.size();
  
  return mean;
}



std::vector<float> KMeans::optimizedCalculateCentroid(const std::vector<std::vector<float>>& vectors) {
  size_t number_of_vectors = vectors.size();
  
  if (number_of_vectors == 0) {
    return std::vector<float>(vectorspace_dimension, 0);
  }
  
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

    for (; j < vectorspace_dimension; j++)
      mean[j] += vectors[i][j];
  }

  size_t i = 0;
  float32x4_t inv_num_vectors = vdupq_n_f32(1.0f / number_of_vectors);
  
  for (; i + 7 < vectorspace_dimension; i += 8) {
    float32x4x2_t simd_mean = vld1q_f32_x2(&mean[i]);
    simd_mean.val[0] = vmulq_f32(simd_mean.val[0], inv_num_vectors);
    simd_mean.val[1] = vmulq_f32(simd_mean.val[1], inv_num_vectors);

    vst1q_f32_x2(&mean[i], simd_mean);
  }

  for (; i < vectorspace_dimension; i++)
    mean[i] *= (1.0f / number_of_vectors);
  
  return mean;
}




std::vector<float> KMeans::optimizedCalculateCentroidFromIndexes(const std::vector<size_t> &vectors_indexes) {
  size_t number_of_vectors = vectors_indexes.size();
  
  if (number_of_vectors == 0) {
    return std::vector<float>(vectorspace_dimension, 0);
  }
  
  std::vector<float> mean(vectorspace_dimension, 0);

  for (size_t i = 0; i < number_of_vectors; i++) {
    size_t j = 0;
    
    for (; j + 7 < vectorspace_dimension; j += 8) {
      float32x4x2_t simd_v = vld1q_f32_x2(&images[vectors_indexes[i]][j]);
      float32x4x2_t simd_mean = vld1q_f32_x2(&mean[j]);
      simd_mean.val[0] = vaddq_f32(simd_mean.val[0], simd_v.val[0]);
      simd_mean.val[1] = vaddq_f32(simd_mean.val[1], simd_v.val[1]);
      
      vst1q_f32_x2(&mean[j], simd_mean);
    }

    for (; j < vectorspace_dimension; j++)
      mean[j] += images[vectors_indexes[i]][j];
  }

  size_t i = 0;
  float32x4_t inv_num_vectors = vdupq_n_f32(1.0f / number_of_vectors);
  
  for (; i + 7 < vectorspace_dimension; i += 8) {
    float32x4x2_t simd_mean = vld1q_f32_x2(&mean[i]);
    simd_mean.val[0] = vmulq_f32(simd_mean.val[0], inv_num_vectors);
    simd_mean.val[1] = vmulq_f32(simd_mean.val[1], inv_num_vectors);

    vst1q_f32_x2(&mean[i], simd_mean);
  }

  for (; i < vectorspace_dimension; i++)
    mean[i] *= (1.0f / number_of_vectors);
  
  return mean;
}



float KMeans::distanceFromClosestCentroid(const std::vector<float> &point) {
  float minimum_distance = std::numeric_limits<float>::max();
  
  for(size_t i = 0; i < centroids.size(); i++)
    minimum_distance = std::min(minimum_distance, optimizedEuclideanDistance(point, centroids[i]));
  
  return minimum_distance;
}



size_t KMeans::indexOfClosestCentroid(const std::vector<float> &point) {
  float minimum_distance = std::numeric_limits<float>::max();
  size_t index = 0;
  
  for(size_t i = 0; i < centroids.size(); i++) {
    float distance_from_centroid = optimizedEuclideanDistance(point, centroids[i]);
    
    if (distance_from_centroid < minimum_distance) {
      minimum_distance = distance_from_centroid;
      index = i;
    }
  }
  
  return index;
}



std::vector<size_t> KMeans::returnClusterElementsIndexes(const size_t &cluster) {
  std::vector<size_t> indexes;
  
  for(size_t i = 0; i < clusters.size(); i++)
    if (clusters[i] == cluster)
      indexes.push_back(i);
  
  return indexes;
}



std::vector<size_t> KMeans::returnLabelElementsIndexes(const size_t &label) {
  std::vector<size_t> indexes;
  
  for(size_t i = 0; i < labels.size(); i++)
    if (labels[i] == label)
      indexes.push_back(i);
  
  return indexes;
}



size_t KMeans::returnNumberOfLabelElements(const size_t &label) {
  size_t counter = 0;
  
  for(size_t i = 0; i < labels.size(); i++)
    if(labels[i] == label)
      counter++;
  
  return counter;
}




float KMeans::clusteringEntropy() {
  size_t N = images.size();
  float entropy = 0.0f;

  for (size_t i = 0; i < centroids.size(); i++) {
    size_t count = returnClusterElementsIndexes(i).size();
    if (count == 0) continue;

    float p = float(count) / float(N);
    entropy -= p * std::log2(p);
  }
  return entropy;
}



float KMeans::trueLabelsEntropy() {
  size_t N = images.size();
  float entropy = 0.0f;
  std::set<size_t> distinct_labels(labels.begin(), labels.end());

  for (size_t i = 0; i < distinct_labels.size(); i++) {
    size_t count = returnNumberOfLabelElements(i);
    if (count == 0) continue;

    float p = float(count) / float(N);
    entropy -= p * std::log2(p);
  }
  return entropy;
}


/// **Unsupervided Measure, a.k.a intertia**
/// Note: by accesing by cluster, since we are jumping from one image to another, we are not taking into account
/// cache locality, and we and up with a lot of cache misses, which results in 11.000ms just the execution of
/// clusteringError(). By accessing by image, we avoid this overhead.
double KMeans::clusteringError() {
  double E = 0.0f;
  
  for (size_t i = 0; i < images.size(); i++)
    E += optimizedEuclideanDistance(images[i], centroids[clusters[i]]);
  
  return E;
}


/// **Supervised Measure**
float KMeans::normalizedMutualInformation() {
  std::set<size_t> distinct_labels(labels.begin(), labels.end());
  
  float HC = clusteringEntropy();
  float HL = trueLabelsEntropy();
  
  float mutual_information = 0.0f;
  float N = float(images.size());
  
  for(int i = 0; i < centroids.size(); i++) {
    for (int j = 0; j < distinct_labels.size(); j++) {
      std::vector<size_t> C_i = returnClusterElementsIndexes(i);
      std::vector<size_t> L_j = returnLabelElementsIndexes(j);
      std::set<size_t> intersection;
      
      std::set_intersection(
                            C_i.begin(), C_i.end(),
                            L_j.begin(), L_j.end(),
                            std::inserter(intersection, intersection.begin())
                            );
      
      if(intersection.size() == 0 || C_i.size() == 0 || L_j.size() == 0) continue;;
      
      mutual_information += (intersection.size() / N) * std::log2((N * intersection.size()) / (C_i.size() * L_j.size()));
    }
  }
  
  return mutual_information / ((HC + HL) / 2.0f);
}



/**===----------------------------------------------------------------------===

 - Clustering Functions
 - kmeans_pp
 - assignmentStep
 
===----------------------------------------------------------------------=== **/


// Initialize the centroids using k-means++
void KMeans::kmeans_pp() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> uni(0, images.size() - 1);
  
  size_t first = uni(gen);
  centroids.push_back(images[first]);
  std::cout << first << std::endl;
  
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


void KMeans::deterministic_initialization() {
  centroids.push_back(images[12354]);
  centroids.push_back(images[15549]);
  centroids.push_back(images[494]);
  centroids.push_back(images[16792]);
  centroids.push_back(images[66488]);
  centroids.push_back(images[12359]);
  centroids.push_back(images[33808]);
  centroids.push_back(images[27472]);
  centroids.push_back(images[7883]);
  centroids.push_back(images[38153]);
}



void KMeans::assignmentStep() {
#pragma omp parallel for schedule(static)
  for(size_t i = 0; i < images.size(); i++)
    clusters[i] = indexOfClosestCentroid(images[i]);
}



void KMeans::updateStep() {
#pragma omp parallel for schedule(static)
  for(size_t i = 0; i < centroids.size(); i++)
    centroids[i] = optimizedCalculateCentroidFromIndexes(returnClusterElementsIndexes(i));
}

void dummyTask() {}

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
  }
  
  {
    Timer timer("Optimized centroid calculation test");
    std::vector<float> v = optimizedCalculateCentroid(images);
  }
  
  {
    Timer timer("Thread Spawning test");
    std::thread t(dummyTask);
    t.join();
  }
  
  deterministic_initialization();
  
// #define DEBUG_KMEANS
  
  double E = 0.0f;
  int i = 0;
  
  {
    Timer timer("Clustering");
    while (true){
      
      {
#ifdef DEBUG_KMEANS
        Timer timer("assignmentStep");
#endif
        assignmentStep();
      }
      
      {
#ifdef DEBUG_KMEANS
        Timer timer("updateStep");
#endif
        updateStep();
      }
      
      {
#ifdef DEBUG_KMEANS
        Timer timer("clusteringError");
#endif
        double E_aux = clusteringError();
        double delta = E_aux - E;
        E = E_aux;
        i++;
        
        if (delta == 0) break;
        
        //std::cout << "Error " << E << std::endl;
        std::cout << "Error Delta: " << delta << " NMI: " << normalizedMutualInformation() << std::endl;
      }
#ifdef DEBUG_KMEANS
      std::cout << std::endl;
#endif
    }
  }
  
  std::cout << "Number of iterations: " << i << std::endl;
}

