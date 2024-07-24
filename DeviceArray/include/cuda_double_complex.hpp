//
// Created by MiseTy on 2023/3/21.
//

#ifndef CSFD_SLAM_NEW_CUDA_DOUBLE_COMPLEX_HPP
#define CSFD_SLAM_NEW_CUDA_DOUBLE_COMPLEX_HPP
#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif
#include <cmath>
#include <cuda/std/complex>
#include <cuda_complex.hpp>

template<class Tp>
class d_complex {
public:
    //    typedef cuda::std::complex<Tp> SingleComplex;
    typedef complex<Tp> SingleComplex;
    typedef Tp value_type;

private:
    SingleComplex re_;
    SingleComplex im_;

public:
    CUDA_CALLABLE_MEMBER
    explicit d_complex(value_type re_re = 0, value_type re_im = 0, value_type im_re = 0, value_type im_im = 0) {
        re_ = SingleComplex(re_re, re_im);
        im_ = SingleComplex(im_re, im_im);
    }
    CUDA_CALLABLE_MEMBER
    explicit d_complex(SingleComplex re, SingleComplex im)
        : re_(re), im_(im) {}
    CUDA_CALLABLE_MEMBER
    explicit d_complex(SingleComplex re) {
        re_ = re;
        im_ = 0;
    }

    CUDA_CALLABLE_MEMBER SingleComplex real() const { return re_; }
    CUDA_CALLABLE_MEMBER SingleComplex imag() const { return im_; }
    CUDA_CALLABLE_MEMBER SingleComplex norm() const {
        return re_ * re_ + im_ * im_;
    }
    CUDA_CALLABLE_MEMBER Tp value() const {
        return re_.real();
    }
    CUDA_CALLABLE_MEMBER Tp grad() const {
        return re_.imag();
    }
    CUDA_CALLABLE_MEMBER Tp hessian() const {
        return im_.imag();
    }
    CUDA_CALLABLE_MEMBER void real(SingleComplex re) { re_ = re; }
    CUDA_CALLABLE_MEMBER void imag(SingleComplex im) { im_ = im; }

    CUDA_CALLABLE_MEMBER d_complex &operator=(const value_type &re) {
        re = re;
        im_ = 0;
        return *this;
    }

    CUDA_CALLABLE_MEMBER d_complex &operator=(const SingleComplex &re) {
        re = re;
        im_ = 0;
        return *this;
    }

    CUDA_CALLABLE_MEMBER d_complex &operator+=(const value_type &re) {
        re_ += re;
        return *this;
    }
    CUDA_CALLABLE_MEMBER d_complex &operator-=(const value_type &re) {
        re_ -= re;
        return *this;
    }
    CUDA_CALLABLE_MEMBER d_complex &operator*=(const value_type &re) {
        re_ *= re;
        im_ *= re;
        return *this;
    }
    CUDA_CALLABLE_MEMBER d_complex &operator/=(const value_type &re) {
        re_ /= re;
        im_ /= re;
        return *this;
    }

    CUDA_CALLABLE_MEMBER d_complex &operator+=(const SingleComplex &re) {
        re_ += re;
        return *this;
    }
    CUDA_CALLABLE_MEMBER d_complex &operator-=(const SingleComplex &re) {
        re_ -= re;
        return *this;
    }
    CUDA_CALLABLE_MEMBER d_complex &operator*=(const SingleComplex &re) {
        re_ *= re;
        im_ *= re;
        return *this;
    }
    CUDA_CALLABLE_MEMBER d_complex &operator/=(const SingleComplex &re) {
        re_ /= re;
        im_ /= re;
        return *this;
    }

    CUDA_CALLABLE_MEMBER d_complex &operator+=(const d_complex &other) {
        re_ += other.re_;
        im_ += other.im_;
        return *this;
    }
    CUDA_CALLABLE_MEMBER d_complex &operator-=(const d_complex &other) {
        re_ -= other.re_;
        im_ -= other.im_;
        return *this;
    }
    CUDA_CALLABLE_MEMBER d_complex &operator*=(const d_complex &other) {
        SingleComplex real = this->re_ * other.re_ - this->im_ * other.im_;
        SingleComplex imag = this->im_ * other.re_ + this->re_ * other.im_;
        this->re_ = real;
        this->im_ = imag;
        return *this;
    }
    CUDA_CALLABLE_MEMBER d_complex &operator/=(const d_complex &other) {
        const SingleComplex r = this->re_ * other.re_ + this->im_ * other.im_;
        //        printf("aaa(%f, %f)\n", r.real(), r.imag());
        const SingleComplex n = other.norm();
        im_ = (im_ * other.re_ - re_ * other.im_) / n;
        re_ = r / n;
        return *this;
    }
};

// operators:
template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator+(const d_complex<Tp> &lhs) {
    return lhs;
}

template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator-(const d_complex<Tp> &lhs) {
    d_complex<Tp> result(-lhs.real(), -lhs.imag());
    return result;
}


template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator+(const d_complex<Tp> &lhs, const Tp &rhs) {
    d_complex<Tp> result(lhs);
    result += rhs;
    return result;
}

template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator-(const d_complex<Tp> &lhs, const Tp &rhs) {
    d_complex<Tp> result(lhs);
    result -= rhs;
    return result;
}

template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator*(const d_complex<Tp> &lhs, const Tp &rhs) {
    d_complex<Tp> result(lhs);
    result *= rhs;
    return result;
}

template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator/(const d_complex<Tp> &lhs, const Tp &rhs) {
    d_complex<Tp> result(lhs);
    result /= rhs;
    return result;
}
/////////////////////////////////////////////////////////
template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator-(const Tp &lhs, const d_complex<Tp> &rhs) {
    d_complex<Tp> result(-rhs);
    result += lhs;
    return result;
}
/////////////////////////////////////////////////////////

template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator+(const d_complex<Tp> &lhs, const d_complex<Tp> &rhs) {
    d_complex<Tp> result(lhs);
    result += rhs;
    return result;
}

template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator-(const d_complex<Tp> &lhs, const d_complex<Tp> &rhs) {
    d_complex<Tp> result(lhs);
    result -= rhs;
    return result;
}
template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator*(const d_complex<Tp> &lhs, const d_complex<Tp> &rhs) {
    d_complex<Tp> result(lhs);
    result *= rhs;
    return result;
}
template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        operator/(const d_complex<Tp> &lhs, const d_complex<Tp> &rhs) {
    d_complex<Tp> result(lhs);
    result /= rhs;
    return result;
}
////////////////////////////////////////////////////
template<class Tp>
inline CUDA_CALLABLE_MEMBER
        complex<Tp>
        abs(const d_complex<Tp> &x) {
    complex<Tp> temp = x.real() * x.real() + x.imag() * x.imag();
    return sqrt(temp);
}


template<class Tp>
inline CUDA_CALLABLE_MEMBER
        d_complex<Tp>
        sqrt(const d_complex<Tp> &x) {
    d_complex<Tp> result = x;
    complex<Tp> r = abs(x);//|z|
    complex<Tp> sqrt_r = sqrt(r);
    result.real(result.real() + r);  //z+r
    complex<Tp> zrnorm = abs(result);//|z+r|

    if (fabs(zrnorm.real()) < static_cast<Tp>(1e-20) && fabs(zrnorm.imag()) < static_cast<Tp>(1e-20)) {
        result *= sqrt_r;
        return result;
    } else {
        complex<Tp> scale = sqrt_r / zrnorm;
        result *= scale;
        return result;
    }
}
#endif//CSFD_SLAM_NEW_CUDA_DOUBLE_COMPLEX_HPP
