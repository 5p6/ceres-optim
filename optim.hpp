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

    struct AutoDiffCost
    {

    public:
        AutoDiffCost(Vec2 _x, V _y) : x(_x), y(_y) {}

        template <typename T>
        bool operator()(
            const T *const coe,
            T *residuals) const
        {
            const T &a = (T)coe[0];
            const T &b = (T)coe[1];
            T observe_y = exp(a * x(0) * x(0) + b * x(1) * x(1));
            T real_y = (T)y(0);
            residuals[0] = real_y - observe_y;
            return true;
        }
        static ceres::CostFunction *create(const Vec2 x,
                                           const V y)
        {
            return new ceres::AutoDiffCostFunction<Mine::AutoDiffCost, 1, 4>(
                new Mine::AutoDiffCost(x, y));
        }
        Vec2 x;
        V y;
    };

    struct Costfunc
    {

    public:
        template <typename T>
        bool operator()(
            const T *const coe,
            T *residuals) const
        {
            const T a = (T)coe[0];
            const T b = (T)coe[1];
            const T c = (T)coe[2];
            residuals[0] = cos(a) + sin(c);
            return true;
        }
    };

    struct MatOptim
    {
        MatOptim(Vec3 _x,Vec3 _y) :x(_x),y(_y) {}

        template <typename T>
        bool operator()(const T *const K,T *residuals) const
        {
            const T& fx = K[0];
            const T& ux = K[1];
            const T& fy = K[2];
            const T& vy = K[3];

            T u = fx * x(0) + ux * x(2);
            T v = fy * x(1) + vy * x(2);
            T w = (T)x(2);
            residuals[0] = u - y(0);
            residuals[1] = v - y(1);
            residuals[2] = w - y(2);
            return true;
        }
        static ceres::CostFunction* create(Vec3 x,Vec3 y){
            return new ceres::AutoDiffCostFunction<MatOptim, 3, 4>
            (new Mine::MatOptim(x,y));
        }
    private:
        Vec3 x;
        Vec3 y;
    };
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


    struct expCost
    {
    public:
        expCost(double _x, double _y) : x(_x), y(_y) {}

        template <typename T>
        bool operator()(
            const T *const coe,
            T *residuals) const
        {
            if(ceres::isnan(y)) return false;
            const T copy_x = (T) x;
            T observe_y = 
            coe[0] * x* x *x *x  +   coe[1] * x* x *x +
            coe[2] * x* x  +  coe[3] * x
            + coe[4]  ;
            T real_y = (T) y;
            residuals[0] = real_y - observe_y;
            return true;
        }
        static ceres::CostFunction* create(const double x,
                                           const double y)
        {
            return new ceres::AutoDiffCostFunction<Mine::expCost, 1, 5>(
                new Mine::expCost(x, y));
        }
        double x;
        double y;
    };

}

#endif
