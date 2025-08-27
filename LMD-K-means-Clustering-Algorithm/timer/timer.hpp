//
//  timer.hpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/17/25.
//

#ifndef timer_hpp
#define timer_hpp

#include <iostream>
#include <chrono>
#include <string>
#include <unordered_map>

struct Timer {
public:
  Timer();
  Timer(std::string name);
  ~Timer();
  static void printAverages();
  
private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
  void stop();
  std::string name;
  
  static inline std::unordered_map<std::string, std::pair<long long, int>> timings;
};

#endif /* timer_hpp */
