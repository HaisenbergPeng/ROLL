#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>

#include<boost/optional.hpp>
// TODO...........................................
class UnaryFactor: public NoiseModelFactor1<Pose3> {
  // The factor will hold a measurement consisting of an (X,Y) location
  // We could this with a Point2 but here we just use two doubles
  float mx_; // how to init it with float array?
  float my_;
  float mz_;

 public:
  /// shorthand for a smart pointer to a factor
  typedef boost::shared_ptr<UnaryFactor> shared_ptr;

  // The constructor requires the variable key, the (X, Y) measurement value, and the noise model
  UnaryFactor(Key j, float x, float y, float z, const SharedNoiseModel& model):
    NoiseModelFactor1<Pose2>(model, j), mx_(x), my_(y), mz_(z)){}

  ~UnaryFactor() override {}

  // Using the NoiseModelFactor1 base class there are two functions that must be overridden.
  // The first is the 'evaluateError' function. This function implements the desired measurement
  // function, returning a vector of errors when evaluated at the provided variable value. It
  // must also calculate the Jacobians for this measurement function, if requested.
  Vector evaluateError(const Point3& qGuess, boost::optional<gtsam::Matrix&> H = boost::none) const override 
  {
    // how to extend it to 3d???
    // The measurement function for a GPS-like measurement h(q) which predicts the measurement (m) is h(q) = q, q = [qx qy qtheta]
    // The error is then simply calculated as E(q) = h(q) - m:
    // error_x = q.x - mx
    // error_y = q.y - my
    // Node's orientation reflects in the Jacobian, in tangent space this is equal to the right-hand rule rotation matrix
    // H =  [ cos(q.theta)  -sin(q.theta) 0 ]
    //      [ sin(q.theta)   cos(q.theta) 0 ]
    const Rot3& R = q.rotation();
    const Point3& t = q.translation();
    if (H) (*H) = (gtsam::Matrix(6, 6) << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                          0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0).finished();
    return (Vector(3) << t.x()-mx_, t.y() - my_, t.z() - mz_).finished();
  }

  // The second is a 'clone' function that allows the factor to be copied. Under most
  // circumstances, the following code that employs the default copy constructor should
  // work fine.
  gtsam::NonlinearFactor::shared_ptr clone() const override 
  {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new UnaryFactor(*this))); 
  }

  // Additionally, we encourage you the use of unit testing your custom factors,
  // (as all GTSAM factors are), in which you would need an equals and print, to satisfy the
  // GTSAM_CONCEPT_TESTABLE_INST(T) defined in Testable.h, but these are not needed below.
};  // UnaryFactor
