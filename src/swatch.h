#include <chrono>

struct Timer {
  Timer() { reset(); }

  void reset() {
    startTime = std::chrono::high_resolution_clock::now();
  }

  double time() const {
    auto endTime = std::chrono::high_resolution_clock::now();
    return double(std::chrono::duration_cast<std::chrono::microseconds>( endTime - startTime ).count())/1000./1000.;
  }

  std::chrono::high_resolution_clock::time_point startTime;
};
