//
//  kmeans.hpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/17/25.
//

#ifndef kmeans_hpp
#define kmeans_hpp

#include <iostream>
#include <vector>
#include <cmath>
#include <arm_neon.h>

class KMeans {
private:
  /// Points to clusters
  std::vector<std::vector<float>> images;
  std::vector<int> labels;
  
  /// Clustering varibles
  std::vector<size_t> clusters;
  std::vector<std::vector<float>> centroids;
  size_t number_of_centroids;
  size_t vectorspace_dimension;
  
  /// Utility Functions
  float euclideanDistance(std::vector<float> v1, std::vector<float> v2);
  float optimizedEuclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2);
  std::vector<float> calculateCentroid(const std::vector< std::vector<float>>& vectors);
  std::vector<float> optimizedCalculateCentroid(const std::vector< std::vector<float>>& vectors);
  std::vector<float> optimizedCalculateCentroidFromIndexes(const std::vector<size_t> &vectors_indexes);
  float distanceFromClosestCentroid(const std::vector<float> &point);
  size_t indexOfClosestCentroid(const std::vector<float> &point);
  std::vector<size_t> returnCluterElemetsIndexes(const size_t &cluster);
  float clusteringError();
  
  /// Clustering Fuctions
  void kmeans_pp();
  void assignmentStep();
  void updateStep();
  
public:
  KMeans(const std::vector<std::vector<float>> &images,
         const std::vector<int> &_labels,
         size_t _number_of_centroids,
         size_t _vectorspace_dimension);
  void test();
};

#endif /* kmeans_hpp */
