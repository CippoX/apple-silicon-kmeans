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
  std::vector<std::vector<float>> centroids;
  size_t number_of_centroids;
  size_t vectorspace_dimension;
  
  void kmeans_pp();
  float euclideanDistance(std::vector<float> v1, std::vector<float> v2);
  float optimizedEuclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2);
  std::vector<float> calculateCentroid(const std::vector< std::vector<float> >& vectors);
  std::vector<float> optimizedCalculateCentroid(const std::vector< std::vector<float> >& vectors);
  
public:
  KMeans(size_t k, size_t _vectorspace_dimension);
  void test();
};

#endif /* kmeans_hpp */
