#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Key.h> 
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

using namespace gtsam;
using namespace std;

// Localization class to model "GPS-like" measurements
class Localization: public NoiseModelFactor1<Pose2> {
  double mx_, my_; ///< X and Y measurements

public:
  Localization(Key j, double x, double y, const SharedNoiseModel& model):
    NoiseModelFactor1<Pose2>(model, j), mx_(x), my_(y) {}

  Vector evaluateError(const Pose2& q,
                       boost::optional<Matrix&> H = boost::none) const
  {
    const Rot2& R = q.rotation();
    if (H) (*H) = (gtsam::Matrix(2, 3) <<
            R.c(), -R.s(), 0.0,R.s(), R.c(), 0.0).finished();
    return (Vector(2) << q.x() - mx_, q.y() - my_).finished();
  }
};

int main(int argc, char** argv) {
    (void)argc; (void)argv; // Suppress unused parameter warnings

    // 1. Create a factor graph and add factors to it
    NonlinearFactorGraph graph;

    // 2. Adding odometry factors
    auto odomNoise = noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));
    graph.emplace_shared<BetweenFactor<Pose2>>(1, 2, Pose2(2.0, 0.0, 0.0), odomNoise);
    graph.emplace_shared<BetweenFactor<Pose2>>(2, 3, Pose2(0.0, 2.0, 0.0), odomNoise);

    // 3. Adding GPS-like measurements
    auto localNoise = noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.1)); // 10 cm std on x, y
    graph.emplace_shared<Localization>(1, 0.0, 0.0, localNoise);
    graph.emplace_shared<Localization>(2, 0.0, 1.0, localNoise);
    graph.emplace_shared<Localization>(3, 2.0, 1.0, localNoise);

    graph.print("Factor graph:\n");

    // 4. Initial Estimate
    Values initialEstimate;
    initialEstimate.insert(1, Pose2(0.5, 0.0, 0.2));
    initialEstimate.insert(2, Pose2(2.3, 0.1, -0.2));
    initialEstimate.insert(3, Pose2(4.1, 0.1, 0.1));
    initialEstimate.print("\nInitial Estimate:\n");

    // 5. Optimizing using Levenberg-Marquardt Optimization
    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
    Values result = optimizer.optimize();
    result.print("Final Result:\n");

    // 6. Calculate and print marginal covariances for all variables
    Marginals marginals(graph, result);
    cout << "x1 covariance:\n" << marginals.marginalCovariance(1) << endl;
    cout << "x2 covariance:\n" << marginals.marginalCovariance(2) << endl;
    cout << "x3 covariance:\n" << marginals.marginalCovariance(3) << endl;

    return 0;
}
