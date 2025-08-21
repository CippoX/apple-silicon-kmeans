//
//  mini-batch-kmeans.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 27/05/25.
//

#include "mini-batch-kmeans.hpp"
#include "../timer/timer.hpp"
#include "../utility/utility.hpp"


MiniBatchKMeans::MiniBatchKMeans(const std::vector< std::vector<float>> &_images,
                                 const std::vector<int> &_labels,
                                 size_t _number_of_centroids,
                                 size_t _vectorspace_dimension,
                                 size_t _mini_batch_size) {
  
  images = _images;
  labels = _labels;
  number_of_centroids = _number_of_centroids;
  vectorspace_dimension = _vectorspace_dimension;
  mini_batch_size = _mini_batch_size;
  clusters = std::vector<size_t>(_images.size(), -1);
  v_x = std::vector<size_t>(number_of_centroids, 0);
}



float MiniBatchKMeans::optimizedEuclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
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



std::vector<float> MiniBatchKMeans::optimizedCalculateCentroidFromIndexes(const std::vector<size_t> &vectors_indexes) {
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



float MiniBatchKMeans::distanceFromClosestCentroid(const std::vector<float> &point) {
  float minimum_distance = std::numeric_limits<float>::max();
  
  for(size_t i = 0; i < centroids.size(); i++)
    minimum_distance = std::min(minimum_distance, optimizedEuclideanDistance(point, centroids[i]));
  
  return minimum_distance;
}



size_t MiniBatchKMeans::indexOfClosestCentroid(const std::vector<float> &point) {
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



std::vector<size_t> MiniBatchKMeans::returnClusterElementsIndexes(const size_t &cluster) {
  std::vector<size_t> indexes;
  
  for(size_t i = 0; i < clusters.size(); i++)
    if (clusters[i] == cluster)
      indexes.push_back(i);
  
  return indexes;
}



std::vector<size_t> MiniBatchKMeans::returnMiniBatchClusterElementsIndexes(const size_t &cluster) {
  std::vector<size_t> indexes;
  for (size_t idx : mini_batch) {
    if (clusters[idx] == cluster) {
      indexes.push_back(idx);
    }
  }
  return indexes;
}



std::vector<size_t> MiniBatchKMeans::returnLabelElementsIndexes(const size_t &label) {
  std::vector<size_t> indexes;
  
  for(size_t i = 0; i < labels.size(); i++)
    if (labels[i] == label)
      indexes.push_back(i);
  
  return indexes;
}



size_t MiniBatchKMeans::returnNumberOfLabelElements(const size_t &label) {
  size_t counter = 0;
  
  for(size_t i = 0; i < labels.size(); i++)
    if(labels[i] == label)
      counter++;
  
  return counter;
}




float MiniBatchKMeans::clusteringEntropy() {
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



float MiniBatchKMeans::trueLabelsEntropy() {
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
double MiniBatchKMeans::clusteringError() {
  double E = 0.0f;
  
  for (size_t i = 0; i < images.size(); i++)
    E += optimizedEuclideanDistance(images[i], centroids[clusters[i]]);
  
  return E;
}


/// **Supervised Measure**
float MiniBatchKMeans::normalizedMutualInformation() {
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



void MiniBatchKMeans::select_mini_batch(size_t k) {
  std::vector<size_t> aux_vector = std::vector<size_t>(images.size(), 0);
  std::random_device rd;
  std::mt19937 gen(rd());
  
  for(size_t i = 0; i < images.size(); i++)
    aux_vector[i] = i;
  
  std::shuffle(aux_vector.begin(), aux_vector.end(), gen);
  aux_vector.resize(k);
  
  mini_batch = aux_vector;
}

/**===----------------------------------------------------------------------===

 - Clustering Functions
 - kmeans_pp
 - assignmentStep
 
===----------------------------------------------------------------------=== **/


// Initialize the centroids using k-means++
void MiniBatchKMeans::kmeans_pp() {
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



void MiniBatchKMeans::assignWholeDataset() {
  for(size_t i = 0; i < images.size(); i++)
    clusters[i] = indexOfClosestCentroid(images[i]);
}



void MiniBatchKMeans::assignmentStep() {
  for(size_t i = 0; i < mini_batch.size(); i++)
    clusters[mini_batch[i]] = indexOfClosestCentroid(images[mini_batch[i]]);
}



void MiniBatchKMeans::updateStep() {
  for (size_t i = 0; i < centroids.size(); ++i) {
    std::vector<size_t> batch_idxs = returnMiniBatchClusterElementsIndexes(i);
    size_t b_i = batch_idxs.size();
    
    if (b_i == 0) continue;
    v_x[i] += b_i;
    
    float eta = float(b_i) / float(v_x[i]);
    std::vector<float> batch_mean = optimizedCalculateCentroidFromIndexes(batch_idxs);
    
    centroids[i] = (1.0f - eta) * centroids[i] + eta  * batch_mean;
  }
}



void MiniBatchKMeans::test() {
  kmeans_pp();
  assignWholeDataset();
  
  double E = std::numeric_limits<double>::max();
  int i = 0;
  
  {
    Timer timer("Clustering");
    while(true) {
      select_mini_batch(mini_batch_size);
      assignmentStep();
      updateStep();
      
      double E_aux = clusteringError();
      double delta = E_aux - E;
      E = E_aux;
      i++;
      
      //FIXME: first iteration results in instant stopping
      if (delta < 10.0 && delta > -10.0) break;
      
      /*std::cout << "Error " << E << std::endl;
      std::cout << "Error Delta: " << delta << std::endl; */
      
      std::cout << "Error Delta: " << delta << " NMI: " << normalizedMutualInformation() << std::endl;
    }
  }
  
  assignWholeDataset();
  
  std::cout << "Number of iterations: " << i << std::endl;
}
