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
  try {
    if (v1.size() != v2.size()){
      throw KMeansException("The two vectors must have the same dimension.");
    }
  } catch(KMeansException& e) {
    std::cout<<e.what()<<std::endl;
  }
  
  float sum = 0.;
  
  for(int i = 0; i < v1.size(); i++) {
    sum += std::pow(v1[i] - v2[i], 2);
  }
  
  return std::sqrt(sum);
}
