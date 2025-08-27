//
//  timer.cpp
//  LMD-K-means-Clustering-Algorithm
//
//  Created by Tommaso Palmisano on 3/17/25.
//

// timer.cpp
#include "timer.hpp"

Timer::Timer() {
  m_StartTimepoint = std::chrono::high_resolution_clock::now();
  name = "";
}

Timer::Timer(std::string timerName) {
  m_StartTimepoint = std::chrono::high_resolution_clock::now();
  name = timerName;
}

Timer::~Timer() {
  stop();
}

void Timer::stop() {
  auto endTimepoint = std::chrono::high_resolution_clock::now();
  
  auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_StartTimepoint).time_since_epoch().count();
  auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endTimepoint).time_since_epoch().count();
  
  auto duration = end - start;
  
  timings[name].first += duration;
  timings[name].second++;

  auto ms = duration * 0.000001;
  std::cout << name << ": " << ms << " ms\n";
}

void Timer::printAverages() {
  for(const auto& [key, val] : timings) {
    double avg_ns = static_cast<double>(val.first) / val.second;
    double avg_ms = avg_ns * 0.000001;
    if(key.empty()) {
      std::cout << "Average: " << avg_ms << " ms\n";
    } else {
      std::cout << key << " average: " << avg_ms << " ms (" << val.second << " runs)\n";
    }
  }
}
