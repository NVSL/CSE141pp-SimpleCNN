#pragma once
#include <cstdint>
#include <iostream>
#include <iomanip>
#include "throw_assert.hpp"

struct gradient_t
{
	float grad;
	float oldgrad;
	gradient_t(): grad(0), oldgrad(0)
	{
	}
	bool operator==(const gradient_t &o) const {
		return (grad == o.grad && oldgrad == o.oldgrad);
	}
	bool operator!=(const gradient_t &o) const {
		return !(*this == o);
	}
};

std::ostream& operator<<(std::ostream& os, const gradient_t & g)
{
	os << std::setw(2) << std::setprecision(2);
	os << "[" << g.grad << ", " << g.oldgrad << "]";
	return os;
}

struct point_t
{
	int x, y, z;
	point_t(int x, int y, int z) : x(x), y(y), z(z){}
	point_t(): x(0), y(0), z(0){}
	bool operator==(const point_t &o) const {
		return (x == o.x && y == o.y && z == o.z);
	}
	bool operator!=(const point_t &o) const {
		return !(*this == o);
	}
};

std::ostream& operator<<(std::ostream& os, const point_t & g)
{
	os << std::setw(2) << std::setprecision(2);
	os << "(" << g.x << ", " << g.y << ", " << g.z << ")";
	return os;
}


using tdsize = point_t;


#ifdef INCLUDE_TESTS
#include "gtest/gtest.h"


namespace CNNTest {
	class CNNTest : public ::testing::Test {
	};

	TEST_F(CNNTest, gradient_operators) {

		gradient_t t1;
		gradient_t t2;
		gradient_t t3;
		
		t1.oldgrad = 1.42;
		t1.grad = 1.30;

		t2.oldgrad = 3.42;
		t2.grad = 5.30;

		EXPECT_EQ(t1,t1);
		EXPECT_NE(t1,t2);
	}
	
	TEST_F(CNNTest, point_operators) {
		point_t t1(0, 0, 0);
		point_t t2(1, 1, 1);
		point_t t3(0, 0, 0);

		EXPECT_EQ(t1, t1);
		EXPECT_NE(t1, t2);
		EXPECT_EQ(t3, t1);
		EXPECT_EQ(t1, t3);
	}
		       
}	
#endif
