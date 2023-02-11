
#include<ceres/ceres.h>
#include<iostream>
#include<Eigen/Dense>
#include<opencv2/opencv.hpp>
#include"optim.hpp"

void test2(){
    Mine::Vec3 xpoints[100];
    Mine::Vec3 ypoints[100];

    Eigen::Matrix3d K;
    K<<330,0,150,0,330,200,0,0,1;
    Eigen::Matrix<double,2,2,Eigen::RowMajor> m;
    cv::RNG rng;
    m<<100,100,100,100;
    for(int i = 0 ;i<100;i++){
        double x[3] ={
            rng.uniform(0.1,12.5),
            rng.uniform(0.1,12.5),
            rng.uniform(0.1,12.5)
        };
        double zaos = rng.gaussian(1.0);
        xpoints[i] = Mine::Vec3(x[0] + zaos,x[1]+ 2*zaos,x[2]- zaos);
        Mine::Vec3 temp = K*xpoints[i];
        ypoints[i] = temp;
    }
    ceres::Problem problem;
    for(int i = 0;i<100;i++){
        problem.AddResidualBlock(
            Mine::MatOptim::create(xpoints[i],ypoints[i]),
            nullptr,
            m.data()
        );
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 200;
    options.minimizer_progress_to_stdout = true;
    options.gradient_tolerance = 1e-20;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    std::cout<<summary.BriefReport()<<std::endl;
    std::cout<< m <<std::endl;
}
void test10(){
    Mine::Vec3 xpoints[100];
    Mine::Vec3 ypoints[100];

    Eigen::Matrix<double,3,3,Eigen::RowMajor> K;
    K<<330,0,150,0,330,200,0,0,1;
    Eigen::Matrix<double,3,3,Eigen::RowMajor> m;
    cv::RNG rng;
    m<<100,0,100,0,100,100,0,0,1;
    for(int i = 0 ;i<100;i++){
        double x[3] ={
            rng.uniform(0.1,12.5),
            rng.uniform(0.1,12.5),
            rng.uniform(0.1,12.5)
        };
        double zaos = rng.gaussian(1.0);
        xpoints[i] = Mine::Vec3(x[0],x[1],x[2]);
        Mine::Vec3 temp = K*(xpoints[i]+ Mine::Vec3(zaos,2*zaos,-zaos));
        ypoints[i] = temp;
    }
    ceres::Problem problem;
    for(int i = 0;i<100;i++){
        problem.AddResidualBlock(
            Mine::KOptim::create(xpoints[i],ypoints[i]),
            nullptr,
            m.data()
        );
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 200;
    options.minimizer_progress_to_stdout = true;
    options.gradient_tolerance = 1e-20;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    std::cout<<summary.BriefReport()<<std::endl;
    std::cout<< m <<std::endl;
}

int main(int argc,char* argv[])
{
    google::InitGoogleLogging(argv[0]);
    test10();
    return 1;
}