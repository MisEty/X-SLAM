#include "DoubleComplex.h"

#include <iostream>

#pragma region Member Functions
DoubleComplex::DoubleComplex(SingleComplex real, SingleComplex imag)
{
    real_ = real;
    imag_ = imag;
}

DoubleComplex::DoubleComplex(SingleComplex real)
{
    real_ = real;
    imag_ = 0;
}

DoubleComplex::DoubleComplex(MyFloat real_real, MyFloat real_imag, MyFloat imag_real, MyFloat imag_imag){
    real_ = SingleComplex (real_real, real_imag);
    imag_ = SingleComplex (imag_real, imag_imag);

}


DoubleComplex::DoubleComplex(MyFloat real)
{
    real_ = real;
    imag_ = 0;
}

DoubleComplex::DoubleComplex()
{
    real_ = 0;
    imag_ = 0;
}

DoubleComplex::~DoubleComplex()
{
}

SingleComplex DoubleComplex::real() const
{
    return this->real_;
}

SingleComplex DoubleComplex::imag() const
{
    return this->imag_;
}

void DoubleComplex::real(SingleComplex real)
{
    this->real_ = real;
}

void DoubleComplex::imag(SingleComplex imag)
{
    this->imag_ = imag;
}

void DoubleComplex::addPerturbation()
{
    float h = 1e-6;
    this->real_ = SingleComplex(real_.real(), h);
    this->imag_ = SingleComplex(h, 0);
}

void DoubleComplex::clearPerturbation(){
    this->real_ = SingleComplex(real_.real(), 0);
    this->imag_ = SingleComplex(0, 0);
}

DoubleComplex DoubleComplex::operator-() const
{
    DoubleComplex result(-real_, -imag_);
    return result;
}

DoubleComplex& DoubleComplex::operator=(const SingleComplex& other)
{
    real_ = other;
    imag_ = 0;
    return *this;
}

DoubleComplex& DoubleComplex::operator=(const MyFloat& other)
{
    real_ = other;
    imag_ = 0;
    return *this;
}

DoubleComplex& DoubleComplex::operator+=(const MyFloat& other)
{
    real_ += other;
    return *this;
}

DoubleComplex& DoubleComplex::operator-=(const MyFloat& other)
{
    real_ -= other;
    return *this;
}

DoubleComplex& DoubleComplex::operator*=(const MyFloat& other)
{
    real_ *= other;
    imag_ *= other;
    return *this;
}

DoubleComplex& DoubleComplex::operator/=(const MyFloat& other)
{
    real_ /= other;
    imag_ /= other;
    return *this;
}

DoubleComplex& DoubleComplex::operator+=(const SingleComplex& other)
{
    real_ += other;
    return *this;
}

DoubleComplex& DoubleComplex::operator-=(const SingleComplex& other)
{
    real_ -= other;
    return *this;
}

DoubleComplex& DoubleComplex::operator*=(const SingleComplex& other)
{
    real_ *= other;
    imag_ *= other;
    return *this;
}

DoubleComplex& DoubleComplex::operator/=(const SingleComplex& other)
{
    real_ /= other;
    imag_ /= other;
    return *this;
}

DoubleComplex& DoubleComplex::operator+=(const DoubleComplex& other)
{
    this->real_ += other.real_;
    this->imag_ += other.imag_;
    return *this;
}

DoubleComplex& DoubleComplex::operator-=(const DoubleComplex& other)
{
    this->real_ -= other.real_;
    this->imag_ -= other.imag_;
    return *this;
}

DoubleComplex& DoubleComplex::operator*=(const DoubleComplex& other)
{
    SingleComplex real = this->real_ * other.real_ - this->imag_ * other.imag_;
    SingleComplex imag = this->imag_ * other.real_ + this->real_ * other.imag_;
    this->real_ = real;
    this->imag_ = imag;
    return *this;
}

DoubleComplex& DoubleComplex::operator/=(const DoubleComplex& other)
{
    const SingleComplex r = this->real_ * other.real_ + this->imag_ * other.imag_;
    const SingleComplex n = norm(other);
    imag_ = (imag_ * other.real() - real_ * other.imag()) / n;
    real_ = r / n;
    return *this;
}

DoubleComplex operator+(const DoubleComplex& lhs, const MyFloat& rhs)
{
    DoubleComplex result(lhs);
    result += rhs;
    return result;
}

DoubleComplex operator-(const DoubleComplex& lhs, const MyFloat& rhs)
{
    DoubleComplex result(lhs);
    result -= rhs;
    return result;
}

DoubleComplex operator*(const DoubleComplex& lhs, const MyFloat& rhs)
{
    DoubleComplex result(lhs);
    result *= rhs;
    return result;
}

DoubleComplex operator/(const DoubleComplex& lhs, const MyFloat& rhs)
{
    DoubleComplex result(lhs);
    result /= rhs;
    return result;
}


DoubleComplex operator+(const DoubleComplex& lhs, const DoubleComplex& rhs)
{
    DoubleComplex result(lhs);
    result += rhs;
    return result;
}

DoubleComplex operator-(const DoubleComplex& lhs, const DoubleComplex& rhs)
{
    DoubleComplex result(lhs);
    result -= rhs;
    return result;
}

DoubleComplex operator*(const DoubleComplex& lhs, const DoubleComplex& rhs)
{
    DoubleComplex result(lhs);
    result *= rhs;
    return result;
}

DoubleComplex operator/(const DoubleComplex& lhs, const DoubleComplex& rhs)
{
    DoubleComplex result(lhs);
    result /= rhs;
    return result;
}
#pragma endregion


std::ostream& operator<<(std::ostream& os, DoubleComplex& x)
{
    os << '(' << x.real() << ',' << x.imag() << ')';
    return os;
}

std::ostream& operator<<(std::ostream& os, const DoubleComplex& x)
{
    os << '(' << x.real() << ',' << x.imag() << ')';
    return os;
}

bool operator>(const DoubleComplex& lhs, const DoubleComplex& rhs)
{
    return (lhs.real().real() > rhs.real().real());
}

bool operator>(const DoubleComplex& lhs, const SingleComplex& rhs)
{
    return (lhs.real().real() > rhs.real());
}

bool operator>(const DoubleComplex& lhs, const MyFloat& rhs)
{
    return (lhs.real().real() > rhs);
}

bool operator<(const DoubleComplex& lhs, const DoubleComplex& rhs)
{
    return (lhs.real().real() < rhs.real().real());
}

bool operator<(const DoubleComplex& lhs, const SingleComplex& rhs)
{
    return (lhs.real().real() < rhs.real());
}

bool operator<(const DoubleComplex& lhs, const MyFloat& rhs)
{
    return (lhs.real().real() < rhs);
}

SingleComplex real(const DoubleComplex& x)
{
    return x.real();
}

SingleComplex imag(const DoubleComplex& x)
{
    return x.imag();
}

MyFloat fabs(const DoubleComplex& x)
{
    return fabs(x.real().real());
}

SingleComplex abs(const DoubleComplex& x)
{
    SingleComplex temp = x.real() * x.real() + x.imag() * x.imag();
    return sqrt(temp);
}

DoubleComplex abs_d(const DoubleComplex& x){
    return sqrt(x * x);

}


DoubleComplex abs2(const DoubleComplex& x)
{
    return x * x;
}

SingleComplex norm(const DoubleComplex& x)
{
    return x.real() * x.real() + x.imag() * x.imag();
}

SingleComplex arg(const DoubleComplex& x)
{
    return atan2(x.imag(),  x.real());
}

DoubleComplex conj(const DoubleComplex& x)
{
    return DoubleComplex(x.real(), -x.imag());
}

DoubleComplex polar(const SingleComplex& rho, const SingleComplex& theta)
{
    SingleComplex real = rho * cos(theta);
    SingleComplex imag = rho * sin(theta);
    return DoubleComplex(real, imag);
}

DoubleComplex sqrt(const DoubleComplex& x)
{
    DoubleComplex result = x;
    SingleComplex r = abs(x);		//|z|
    SingleComplex sqrt_r = sqrt(r);
    result += r;	//z+r
    SingleComplex zrnorm = abs(result);	//|z+r|

    if (fabs(zrnorm.real()) < static_cast <MyFloat>(1e-20) && fabs(zrnorm.imag()) < static_cast <MyFloat>(1e-20)) {
        result *= sqrt_r;
        return result;
    }
    else {
        SingleComplex scale = sqrt_r / zrnorm;
        result *= scale;
        return result;
    }
}

DoubleComplex exp(const DoubleComplex& x)
{
    SingleComplex real = exp(x.real()) * cos(x.imag());
    SingleComplex imag = exp(x.real()) * sin(x.imag());
    return DoubleComplex(real, imag);
}

DoubleComplex log(const DoubleComplex& x)
{
    SingleComplex r = abs(x);
    SingleComplex real_ = x.real();
    SingleComplex imag_ = x.imag();
    SingleComplex imag = atan2(imag_, real_);
    SingleComplex real = log(r);
    return DoubleComplex(real, imag);
}

DoubleComplex atanh(const DoubleComplex& x)
{
    SingleComplex a1(1, 0);
    SingleComplex a2(0, 0);
    DoubleComplex a(a1, a2);
    return((log(a + x) - log(a - a)) * static_cast <MyFloat>(0.5));
}

DoubleComplex atan(const DoubleComplex& x)
{
    DoubleComplex r(-x.imag(), x.real());
    r = atanh(r);
    return DoubleComplex(r.imag(), -r.real());
}

SingleComplex atan2(const SingleComplex& y, const SingleComplex& x)
{
    SingleComplex r = x * x + y * y;
    r = sqrt(r);
    if (r > static_cast <MyFloat>(0.0)) {
        r += x;
        r = y / r;
    }
    else {
        r -= x;
        r = r / y;
    }
    r = atan(r);
    r *= static_cast <MyFloat>(2.0);
    return r;
}

DoubleComplex atan2(const DoubleComplex& y, const DoubleComplex& x)
{
    DoubleComplex r = x * x + y * y;
    r = sqrt(r);
    if (r > static_cast <MyFloat>(0.0)) {
        r += x;
        r = y / r;
    }
    else {
        r -= x;
        r = r / y;
    }
    r = atan(r);
    r *= static_cast <MyFloat>(2.0);
    return r;
}

DoubleComplex sin(const DoubleComplex& x)
{
    SingleComplex real = cosh(-x.imag()) * sin(x.real());
    SingleComplex imag = -sinh(-x.imag()) * cos(x.real());
    return DoubleComplex(real, imag);
}

DoubleComplex cos(const DoubleComplex& x)
{
    SingleComplex real = cosh(-x.imag()) * cos(x.real());
    SingleComplex imag = sinh(-x.imag()) * sin(x.real());
    return DoubleComplex(real, imag);
}

DoubleComplex pow(const DoubleComplex& x, const MyFloat y)
{
    DoubleComplex r = log(x);
    r = polar(exp(y * r.real()), y * r.imag());
    return r;
}
