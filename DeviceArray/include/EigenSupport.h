#pragma once
#include "DoubleComplex.h"
#include <Eigen/Core>

namespace Eigen {
    template<> struct NumTraits<DoubleComplex>
        : NumTraits<SingleComplex> // permits to get the epsilon, dummy_precision, lowest, highest functions
    {
        typedef SingleComplex Real;
        typedef DoubleComplex NonInteger;
        typedef DoubleComplex Nested;

        enum {
            IsComplex = 1,
            IsInteger = 0,
            IsSigned = 1,
            RequireInitialization = 1,
            ReadCost = 1,
            AddCost = 3,
            MulCost = 3
        };
    };
}

typedef Eigen::Matrix<DoubleComplex, 4, 1> Vector4dc;
typedef Eigen::Matrix<DoubleComplex, 3, 1> Vector3dc;
typedef Eigen::Matrix<DoubleComplex, 2, 1> Vector2dc;

typedef Eigen::Matrix<DoubleComplex, Eigen::Dynamic, 1> VectorXdc;
typedef Eigen::Matrix<DoubleComplex, 4, 4> Matrix4dc;
typedef Eigen::Matrix<DoubleComplex, 3, 3> Matrix3dc;
typedef Eigen::Matrix<DoubleComplex, Eigen::Dynamic, Eigen::Dynamic> MatrixXdc;