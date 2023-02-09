

#include<ceres/ceres.h>
#include<Eigen/Dense>
#include<iostream>
#include<cmath>
#include<ceres/rotation.h>
class Costfunc{
  public:
  Costfunc(
    Eigen::Vector<double,3> _p1,
    Eigen::Vector<double,2> _p2):p1(_p1),p2(_p2){

    }
    /*
    the matrix is 12 size ,3 for world rotation , 3 for world translation
    2 for image focal length, 2 for image distortion, 2 for image translation
    */
    template<class T>
    bool operator()(
      const T* const matrix,
      T* residuals
      ) const {
        T p[3];
        p[0] = (T)p1(0);
        p[1] = (T)p1(1);
        p[2] = (T)p1(2);
        T point[3];
        // Xc = RX + t 
        //this is Xm = RX 
        ceres::AngleAxisRotatePoint(matrix,p,point);
        // Xc = Xm + t
        point[0] = point[0] + matrix[3];
        point[1] = point[1] + matrix[4];
        point[2] = point[2] + matrix[5];
        
        //x = Xc/z
        const T x = point[0]/point[2];
        const T y = point[1]/point[2];
        // dist
        const T l1 = matrix[6];
        const T l2 = matrix[7];
        T r = x*x+y*y;
        T dist  = 1.0 + r*(l1 + l2*r);

        const T focalx = matrix[8];
        const T focaly = matrix[9];
        const T ux = matrix[10];
        const T vy = matrix[11]; 

        T predict_x  = focalx * dist * x + ux;
        T predict_y = focaly * dist * y + vy;
        residuals[0] = predict_x - p2[0];
        residuals[1] = predict_y - p2[1];
        return true;
    }


  private:
  Eigen::Vector<double,3> p1;
  Eigen::Vector<double,2> p2;
};

int main(int argc, char* argv[]){
  google::InitGoogleLogging(argv[0]);
  // data points
  std::vector<Eigen::Vector<double,3>> objpoints;
  std::vector<Eigen::Vector<double,2>> imgpoints;
  for(int i = 0;i<8;i++)
  {
    for(int j  =0;j<11;j++)
    {
      objpoints.push_back(Eigen::Vector<double,3>(j,i,0));
    }
  }
  imgpoints.push_back(Eigen::Vector<double,2>(1909.77,876.49));
  imgpoints.push_back(Eigen::Vector<double,2>(1821.21,874.34));
  imgpoints.push_back(Eigen::Vector<double,2>(1733.04,872.244));
  imgpoints.push_back(Eigen::Vector<double,2>(1645.1,870.194));
  imgpoints.push_back(Eigen::Vector<double,2>(1557.51,868.058));
  imgpoints.push_back(Eigen::Vector<double,2>(1470.14,865.96));
  imgpoints.push_back(Eigen::Vector<double,2>(1383.11,864.019));
  imgpoints.push_back(Eigen::Vector<double,2>(1296.37,861.854));
  imgpoints.push_back(Eigen::Vector<double,2>(1209.98,859.821));
  imgpoints.push_back(Eigen::Vector<double,2>(1124.06,857.769));



  // the matrix set
  Eigen::Vector<double,12> matrix;
  Eigen::Vector<double,3> rotation;
  rotation<<1,2,3;
  Eigen::Vector<double,3> t;
  t<<1,2,3;
  Eigen::Vector<double,2> dist;
  dist<<1,2;
  Eigen::Vector<double,4> K;
  K<<1,2,3,4;
  // set 
  matrix(Eigen::seq(0,2)) = rotation;
  matrix(Eigen::seq(3,5)) = t;
  matrix(Eigen::seq(6,7)) = dist;
  matrix(Eigen::seq(8,11)) = K;


  ceres::Problem problem;
  for(int i =0;i<10;i++){
  problem.AddResidualBlock(
    new ceres::AutoDiffCostFunction<Costfunc,2,12>
    (new Costfunc(objpoints[i],imgpoints[i])),
    nullptr,
    &matrix[0]
  );
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = 100;
  options.minimizer_progress_to_stdout = true;

  ceres::Solver::Summary summary;
  ceres::Solve(options,&problem,&summary);
  std::cout<< summary.BriefReport()<<std::endl;
  return 1;
}
