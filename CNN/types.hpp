#pragma once
#include <cstdint>
#include <iostream>
#include <iomanip>
#include "throw_assert.hpp"


struct gradient_t
{
	/* 
	   gradient_t is a convenience structure for storing the old
	   value of the gradient along with the current value.

	   `oldgrad` is only used for the "momentum" term in the
	   weight updating code (see optimization_method.hpp).

	   `grad` stores the gradient for the current round of back
	   propagation.
	*/
	double grad;
	double oldgrad;
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


struct point_t
{
	/* 
	   point_t stores the coordinates of an item in a tensor_t.
	*/
	
	int x, y, z, b;
	point_t(int x, int y, int z, int b=0) : x(x), y(y), z(z), b(b){}
	point_t(): x(0), y(0), z(0), b(0){}
	bool operator==(const point_t &o) const {
		return (x == o.x && y == o.y && z == o.z && b == o.b);
	}
	bool operator!=(const point_t &o) const {
		return !(*this == o);
	}
};

/* tdsize is an alias for point_t.  It is used to specify the size of
   tensor_t objects. */
using tdsize = point_t;


inline std::ostream& operator<<(std::ostream& os, const gradient_t & g)
{
	os << std::setw(2) << std::setprecision(2);
	os << "[" << g.grad << ", " << g.oldgrad << "]";
	return os;
}

inline std::ostream& operator<<(std::ostream& os, const point_t & g)
{
	os << std::setw(2) << std::setprecision(2);
	os << "(" << g.x << ", " << g.y << ", " << g.z << ", " << g.b << ")";
	return os;
}



#ifdef INCLUDE_TESTS
#include <gtest/gtest.h>


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

