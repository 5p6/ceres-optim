#ifndef OPTIM_HPP
#define OPTIM_HPP
#include <iostream>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <vector>
#include <cmath>

namespace Mine
{

    using Vec3 = Eigen::Vector<double, 3>;
    using Vec2 = Eigen::Vector<double, 2>;
    using V = Eigen::Vector<double, 1>;
    struct KOptim
    {
        KOptim(Vec3 _x,Vec3 _y) :x(_x),y(_y) {}

        template <typename T>
        bool operator()(const T *const K,T *residuals) const
        {
            const T& fx = *K;
            const T& ux = *(K + 2);
            const T& fy = *(K + 4);
            const T& vy = *(K + 5);
            T u = fx * x(0) + ux * x(2);
            T v = fy * x(1) + vy * x(2);
            T w = (T)x(2);
            residuals[0] = u - y(0);
            residuals[1] = v - y(1);
            residuals[2] = w - y(2);
            return true;
        }
        static ceres::CostFunction* create(Vec3 x,Vec3 y){
            return new ceres::AutoDiffCostFunction<KOptim, 3, 9>
            (new Mine::KOptim(x,y));
        }
    private:
        Vec3 x;
        Vec3 y;
    };
#endif
