#include "EigenSupport.h"
#include "cxtimers.h"

#include "Eigen/Eigen"

#include <fstream>
#include <iostream>

DoubleComplex f1(DoubleComplex x, DoubleComplex y)
{
    return (x + y) * (x + y);
}

DoubleComplex f2(Vector3dc x)
{
    return (sin(x[0]) + sin(x[1]) + sin(x[2])) * (sin(x[0]) + sin(x[1]) + sin(x[2]));
}
SingleComplex multiplication_our(const SingleComplex &a, const SingleComplex &b)
{
    MyFloat real = a.real() * b.real();
    MyFloat imag = a.imag() * b.real() + a.real() * b.imag();
    return SingleComplex(real, imag);
}

SingleComplex multiplication_raw(const SingleComplex &a, const SingleComplex &b)
{
    MyFloat real = a.real() * b.real() - a.imag() * b.imag();
    MyFloat imag = a.imag() * b.real() + a.real() * b.imag();
    return SingleComplex(real, imag);
}

SingleComplex division_our(const SingleComplex &a, const SingleComplex &b)
{
    MyFloat real = a.real() / b.real();
    MyFloat imag = (a.imag() * b.real() - a.real() * b.imag()) / (b.real() * b.real() + b.imag() * b.imag());
    return SingleComplex(real, imag);
}

SingleComplex division_raw(const SingleComplex &a, const SingleComplex &b)
{
    MyFloat real = (a.real() * b.real() + a.imag() * b.imag()) / (b.real() * b.real() + b.imag() * b.imag());
    MyFloat imag = (a.imag() * b.real() - a.real() * b.imag()) / (b.real() * b.real() + b.imag() * b.imag());
    return SingleComplex(real, imag);
}

SingleComplex exp_our(const SingleComplex &a)
{
    MyFloat real = exp(a.real());
    MyFloat imag = exp(a.real()) * sin(a.imag());
    return SingleComplex(real, imag);
}

SingleComplex exp_raw(const SingleComplex &a)
{
    MyFloat real = exp(a.real()) * cos(a.imag());
    MyFloat imag = exp(a.real()) * sin(a.imag());
    return SingleComplex(real, imag);
}

SingleComplex sin_our(const SingleComplex &a)
{
    MyFloat real = sin(a.real());
    MyFloat imag = -sinh(-a.imag()) * cos(a.real());
    return SingleComplex(real, imag);
}

SingleComplex sin_raw(const SingleComplex &a)
{
    MyFloat real = sin(a.real()) * cosh(-a.imag());
    MyFloat imag = -sinh(-a.imag()) * cos(a.real());
    return SingleComplex(real, imag);
}

SingleComplex pow_our(const SingleComplex &a, const int &n)
{
    MyFloat real = pow(a.real(), n);
    MyFloat imag = pow(norm(a), n) * sin(n * arg(a));
    return SingleComplex(real, imag);
}

SingleComplex pow_raw(const SingleComplex &a, const int &n)
{
    MyFloat real = pow(norm(a), n) * cos(n * arg(a));
    MyFloat imag = pow(norm(a), n) * sin(n * arg(a));
    return SingleComplex(real, imag);
}

int main()
{
    std::cout << "1. simple test for complex acceleration" << std::endl;
    cx::timer time_logger;
    double total_time = 0;
    float h = 1e-6;
    int launch_number = 1000000;
    SingleComplex a(0.5, h), b(-1.5, h);
    std::cout << "run standard multiplication" << std::endl;
    time_logger.reset();
    for (int i = 0; i < launch_number; i++)
    {
        multiplication_raw(a, b);
    }
    total_time = time_logger.lap_ms();
    printf("mean compute time = %.3f ms\n", total_time);

    std::cout << "run our multiplication" << std::endl;
    time_logger.reset();
    for (int i = 0; i < launch_number; i++)
    {
        multiplication_our(a, b);
    }
    total_time = time_logger.lap_ms();
    printf("mean compute time = %.3f ms\n", total_time);
    std::cout << "value: " << multiplication_our(a, b) << "\t" << a * b << std::endl;

    std::cout << "run standard division" << std::endl;
    time_logger.reset();
    for (int i = 0; i < launch_number; i++)
    {
        division_raw(a, b);
    }
    total_time = time_logger.lap_ms();
    printf("mean compute time = %.3f ms\n", total_time);

    std::cout << "run our division" << std::endl;
    time_logger.reset();
    for (int i = 0; i < launch_number; i++)
    {
        division_our(a, b);
    }
    total_time = time_logger.lap_ms();
    printf("mean compute time = %.3f ms\n", total_time);
    std::cout << "value: " << division_our(a, b) << "\t" << a / b << std::endl;

    std::cout << "run standard exp" << std::endl;
    time_logger.reset();
    for (int i = 0; i < launch_number; i++)
    {
        exp_raw(a + b);
    }
    total_time = time_logger.lap_ms();
    printf("mean compute time = %.3f ms\n", total_time);

    std::cout << "run our exp" << std::endl;
    time_logger.reset();
    for (int i = 0; i < launch_number; i++)
    {
        exp_our(a + b);
    }
    total_time = time_logger.lap_ms();
    printf("mean compute time = %.3f ms\n", total_time);
    std::cout << "value: " << exp_our(a + b) << "\t" << exp(a + b) << std::endl;

    std::cout << "run standard sin" << std::endl;
    time_logger.reset();
    for (int i = 0; i < launch_number; i++)
    {
        sin_raw(a + b);
    }

    total_time = time_logger.lap_ms();
    printf("mean compute time = %.3f ms\n", total_time);

    std::cout << "run our sin" << std::endl;
    time_logger.reset();
    for (int i = 0; i < launch_number; i++)
    {
        sin_our(a + b);
    }
    total_time = time_logger.lap_ms();
    printf("mean compute time = %.3f ms\n", total_time);
    std::cout << "value: " << sin_our(a + b) << "\t" << sin(a + b) << std::endl;

    std::cout << "run standard pow" << std::endl;
    time_logger.reset();
    for (int i = 0; i < launch_number; i++)
    {
        pow_raw(a + b, 3);
    }

    total_time = time_logger.lap_ms();
    printf("mean compute time = %.3f ms\n", total_time);

    std::cout << "run our pow" << std::endl;
    time_logger.reset();
    for (int i = 0; i < launch_number; i++)
    {
        pow_our(a + b, 3);
    }
    total_time = time_logger.lap_ms();
    printf("mean compute time = %.3f ms\n", total_time);
    std::cout << "value: " << pow_our(a + b, 3) << "\t" << pow(a + b, 3) << std::endl;

    // test double complex
    std::cout << "2. test high order chain rule by compute f1(x,y)=(x+y)^2, x=t*t,  y=sin(t)" << std::endl;
    DoubleComplex t(SingleComplex(0.5, h), SingleComplex(h, 0));
    DoubleComplex x = t * t;
    DoubleComplex y = sin(t);
    float part_x_part_t = x.real().imag() / h;
    float part_xx_part_tt = x.imag().imag() / h / h;
    float part_y_part_t = y.real().imag() / h;
    float part_yy_part_tt = y.imag().imag() / h / h;
    auto loss = f1(x, y);
    std::cout << "a. compute gradient and second order differentiation by DCSFD" << std::endl;
    std::cout << "gradient = " << loss.real().imag() / h << std::endl;
    std::cout << "second order differentiation = " << loss.imag().imag() / h / h << std::endl;
    float x_ = x.real().real();
    float y_ = y.real().real();
    float part_f_part_x = f1(DoubleComplex(x_, h, h, 0), DoubleComplex(y_, 0, 0, 0)).real().imag() / h;
    float part_f_part_y = f1(DoubleComplex(x_, 0, 0, 0), DoubleComplex(y_, h, h, 0)).real().imag() / h;
    float part_ff_part_xx = f1(DoubleComplex(x_, h, h, 0), DoubleComplex(y_, 0, 0, 0)).imag().imag() / h / h;
    float part_ff_part_yy = f1(DoubleComplex(x_, 0, 0, 0), DoubleComplex(y_, h, h, 0)).imag().imag() / h / h;
    float part_ff_part_xy = f1(DoubleComplex(x_, h, 0, 0), DoubleComplex(y_, 0, h, 0)).imag().imag() / h / h;
    auto part_f_part_t = part_f_part_x * part_x_part_t + part_f_part_y * part_y_part_t;
    auto part_ff_part_tt = part_f_part_x * part_xx_part_tt + part_f_part_y * part_yy_part_tt +
                           part_x_part_t * part_x_part_t * part_ff_part_xx + part_y_part_t * part_y_part_t * part_ff_part_yy +
                           part_x_part_t * part_y_part_t * (part_ff_part_xy + part_ff_part_xy);
    std::cout << "b. compute gradient and second order differentiation by chain rule" << std::endl;
    std::cout << "gradient = " << part_f_part_t << std::endl;
    std::cout << "second order differentiation = " << part_ff_part_tt << std::endl;
}
