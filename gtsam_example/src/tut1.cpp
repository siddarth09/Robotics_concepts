// To represent robotics position (x,y,theta)
#include <gtsam/geometry/Pose2.h>
// Measurements are represented as 'factors'
#include <gtsam/slam/PriorFactor.h> // Initialize the robot
#include <gtsam/slam/BetweenFactor.h> // To use the relative motion of the robot described by odometry 

// Optimizers
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

// Calculating marginal covariances after optimizing 
#include <gtsam/nonlinear/Marginals.h>

// Nonlinear solvers are iterative solvers in gtsam, they linearize
// a nonlinear function around initial linearization point, then solve the linear systems
// to update the linearization point. 

#include <gtsam/nonlinear/Values.h> // this stores a set of variable values

using namespace gtsam;
using namespace std;

int main(int argc, char** argv) {
    // Create a nonlinear factor graph
    NonlinearFactorGraph graph;

    // Adding a prior on the first pose, which consists of mean and covariance (noise)
    Pose2 priorMean(0.0, 0.0, 0.0);
    auto priorNoise= noiseModel::Diagonal::Sigmas (Vector3(0.3,0.3,0.1));
    graph.addPrior(1,priorMean,priorNoise);

    //Adding odometry factors
    Pose2 odom(2.0,0.0,0.0);
    auto odomNoise= noiseModel::Diagonal::Sigmas (Vector3(0.3,0.3,0.01));
    graph.emplace_shared<BetweenFactor<Pose2> >(1,2,odom,odomNoise);
    Pose2 odom1(2.1,3.0,2.0);
    graph.emplace_shared<BetweenFactor<Pose2> >(2,3,odom1,odomNoise);
    graph.print("Factor Graph:\n");

    Values initial;
    initial.insert(1,Pose2(0.5,0.0,0.2));
    initial.insert(2,Pose2(2.3,0.1,-0.2));
    initial.insert(3,Pose2(4.1,0.1,0.1));
    initial.print("Initial estimate:\n");

    Values result=LevenbergMarquardtOptimizer(graph,initial).optimize();

    result.print("Final Result:\n");

    cout.precision(2);
    Marginals marginals(graph,result);
    cout<<"x1 covariance: \n" << marginals.marginalCovariance(1)<<endl;
    cout<<"x2 covariance: \n" << marginals.marginalCovariance(2)<<endl;
    cout<<"x3 covariance: \n" << marginals.marginalCovariance(3)<<endl;
    return 0;
}
