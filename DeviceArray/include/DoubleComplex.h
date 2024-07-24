#pragma once
#include <complex>
#include <Eigen/Dense>

typedef float MyFloat;
typedef Eigen::Matrix4f Matrix4;
typedef Eigen::Matrix3f Matrix3;
typedef Eigen::MatrixXf MatrixX;
typedef Eigen::Vector4f Vector4;
typedef Eigen::Vector3f Vector3;
typedef Eigen::VectorXf VectorX;


typedef std::complex<MyFloat> SingleComplex;
class DoubleComplex {
private:
    SingleComplex real_;
    SingleComplex imag_;

public:
    DoubleComplex(SingleComplex real, SingleComplex imag);
    DoubleComplex(SingleComplex real);
    DoubleComplex(MyFloat real);
    DoubleComplex(MyFloat real_real, MyFloat real_imag, MyFloat imag_real, MyFloat imag_imag);

    DoubleComplex();
    ~DoubleComplex();
    SingleComplex real() const;
    SingleComplex imag() const;
    void real(SingleComplex real);
    void imag(SingleComplex imag);
    void addPerturbation();
    void clearPerturbation();

    DoubleComplex operator-() const;
    DoubleComplex& operator=(const SingleComplex& other);
    DoubleComplex& operator=(const MyFloat& other);

    DoubleComplex& operator+=(const MyFloat& other);
    DoubleComplex& operator-=(const  MyFloat& other);
    DoubleComplex& operator*=(const  MyFloat& other);
    DoubleComplex& operator/=(const  MyFloat& other);

    DoubleComplex& operator+=(const SingleComplex& other);
    DoubleComplex& operator-=(const SingleComplex& other);
    DoubleComplex& operator*=(const SingleComplex& other);
    DoubleComplex& operator/=(const SingleComplex& other);

    DoubleComplex& operator+=(const DoubleComplex& other);
    DoubleComplex& operator-=(const DoubleComplex& other);
    DoubleComplex& operator*=(const DoubleComplex& other);
    DoubleComplex& operator/=(const DoubleComplex& other);
};

std::ostream& operator<<(std::ostream& os, DoubleComplex& x);
std::ostream& operator<<(std::ostream& os, const DoubleComplex& x);

bool operator>(const DoubleComplex& lhs, const DoubleComplex& rhs);
bool operator>(const DoubleComplex& lhs, const SingleComplex& rhs);
bool operator>(const DoubleComplex& lhs, const MyFloat& rhs);
bool operator<(const DoubleComplex& lhs, const DoubleComplex& rhs);
bool operator<(const DoubleComplex& lhs, const SingleComplex& rhs);
bool operator<(const DoubleComplex& lhs, const MyFloat& rhs);

DoubleComplex operator+(const DoubleComplex& lhs, const MyFloat& rhs);
DoubleComplex operator-(const DoubleComplex& lhs, const MyFloat& rhs);
DoubleComplex operator*(const DoubleComplex& lhs, const MyFloat& rhs);
DoubleComplex operator/(const DoubleComplex& lhs, const MyFloat& rhs);

DoubleComplex operator+(const DoubleComplex& lhs, const DoubleComplex& rhs);
DoubleComplex operator-(const DoubleComplex& lhs, const DoubleComplex& rhs);
DoubleComplex operator*(const DoubleComplex& lhs, const DoubleComplex& rhs);
DoubleComplex operator/(const DoubleComplex& lhs, const DoubleComplex& rhs);

SingleComplex real(const DoubleComplex& x);
SingleComplex imag(const DoubleComplex& x);
MyFloat fabs(const DoubleComplex& x);
SingleComplex abs(const DoubleComplex& x);
DoubleComplex abs_d(const DoubleComplex& x);
DoubleComplex abs2(const DoubleComplex& x);
SingleComplex norm(const DoubleComplex& x);
SingleComplex arg(const DoubleComplex& x);
DoubleComplex conj(const DoubleComplex& x);
DoubleComplex polar(const SingleComplex& rho, const SingleComplex& theta);
DoubleComplex sqrt(const DoubleComplex& x);
DoubleComplex exp(const DoubleComplex& x);
DoubleComplex log(const DoubleComplex& x);

DoubleComplex atanh(const DoubleComplex& x);
DoubleComplex atan(const DoubleComplex& x);
SingleComplex atan2(const SingleComplex& y, const SingleComplex& x);
DoubleComplex atan2(const DoubleComplex& y, const DoubleComplex& x);
DoubleComplex sin(const DoubleComplex& x);
DoubleComplex cos(const DoubleComplex& x);
DoubleComplex pow(const DoubleComplex& x, const MyFloat y);