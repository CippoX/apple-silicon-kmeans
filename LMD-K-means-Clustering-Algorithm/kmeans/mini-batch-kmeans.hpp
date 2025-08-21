//
//  mini-batch-kmeans.hpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 27/05/25.
//

#ifndef minibatchkmeans_hpp
#define minibatchkmeans_hpp

#include <iostream>
#include <vector>
#include <cmath>
#include <arm_neon.h>
#include <set>
#include <random>

class MiniBatchKMeans {
private:
  /// Points to clusters
  std::vector<std::vector<float>> images;
  std::vector<int> labels;
  
  /// Clustering constants & varibles
  size_t number_of_centroids;
  size_t vectorspace_dimension;
  size_t mini_batch_size;

  std::vector<size_t> clusters;
  std::vector<std::vector<float>> centroids;
  std::vector<size_t> mini_batch;
  std::vector<size_t> v_x;
  
  /// Utility Functions
  float optimizedEuclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2);
  std::vector<float> optimizedCalculateCentroidFromIndexes(const std::vector<size_t> &vectors_indexes);
  float distanceFromClosestCentroid(const std::vector<float> &point);
  size_t indexOfClosestCentroid(const std::vector<float> &point);
  std::vector<size_t> returnClusterElementsIndexes(const size_t &cluster);
  std::vector<size_t> returnMiniBatchClusterElementsIndexes(const size_t &cluster);
  std::vector<size_t> returnLabelElementsIndexes(const size_t &label);
  size_t returnNumberOfLabelElements(const size_t &label);
  void select_mini_batch(size_t k);
  
  float clusteringEntropy();
  float trueLabelsEntropy();
  double clusteringError();
  float normalizedMutualInformation();
  
  /// Clustering Fuctions
  void kmeans_pp();
  void assignmentStep();
  void assignWholeDataset();
  void updateStep();
  
public:
  MiniBatchKMeans(const std::vector<std::vector<float>> &images,
                  const std::vector<int> &_labels,
                  size_t _number_of_centroids,
                  size_t _vectorspace_dimension,
                  size_t _mini_batch_size);
  void test();
};

#endif /* kmeans_hpp */
