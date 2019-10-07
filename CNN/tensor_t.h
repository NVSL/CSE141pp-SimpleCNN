#pragma once
#include "types.h"
#include <vector>
#include <string.h>
#include <cmath>


#define EPSILON 1e-8

static float rand_f(int maxval) {
	return 1.0f / maxval * rand() / float( RAND_MAX );
}

template<typename T>
struct tensor_t
{
	T * data;

	tdsize size;

	tensor_t( int _x, int _y, int _z ) : size(_x, _y, _z) {
		throw_assert(size.x > 0 && size.y > 0 && size.z > 0,  "Tensor initialized with negative dimensions");
		data = new T[_x * _y * _z]();
	}

	tensor_t(const tdsize & _size) : size(_size)
	{
		data = new T[_size.x * _size.y * _size.z]();
	}

	tensor_t( const tensor_t& other ) :size(other.size)
	{
		data = new T[other.size.x *other.size.y *other.size.z];
		memcpy(
			this->data,
			other.data,
			other.size.x *other.size.y *other.size.z * sizeof( T )
		);
	}
	
	tensor_t<T> & operator=(const tensor_t& other )
	{
		delete[] data;
		size = other.size;
		data = new T[other.size.x *other.size.y *other.size.z];
		memcpy(
			this->data,
			other.data,
			other.size.x *other.size.y *other.size.z * sizeof( T )
			);
		return *this;
	}

	tensor_t<T> operator+( tensor_t<T>& other )
	{
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			clone.data[i] += other.data[i];
		return clone;
	}

	tensor_t<T> operator-( tensor_t<T>& other )
	{
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			clone.data[i] -= other.data[i];
		return clone;
	}

	T& operator()( int _x, int _y, int _z )
	{
		return this->get( _x, _y, _z );
	}

	const T& operator()( int _x, int _y, int _z ) const
	{
		return this->get( _x, _y, _z );
	}

	bool operator==(const tensor_t<T> & other) const
	{
		if (other.size != this->size)
			return false;
		
		for ( int x = 0; x < this->size.x; x++ )
			for ( int y = 0; y < this->size.y; y++ )
				for ( int z = 0; z < this->size.z; z++ )
					//if (fabs(other(x,y,z) - (*this)(x,y,z)) > EPSILON) 
					if (other(x,y,z) != (*this)(x,y,z))
						return false;
		return true;
	}

	bool operator!=(const tensor_t<T> & o) const {
		return !(*this == o);
	}
	
	T& get( int _x, int _y, int _z )
	{
		throw_assert( _x >= 0 && _y >= 0 && _z >= 0, "Tried to read tensor at negative coordinates" );
		throw_assert( _x < size.x && _y < size.y && _z < size.z, "Tried to read tensor out of bounds" );

		return data[
			_z * (size.x * size.y) +
				_y * (size.x) +
				_x
		];
	}

	const T & get( int _x, int _y, int _z ) const {
		throw_assert( _x >= 0 && _y >= 0 && _z >= 0 , "Tried to read tensor at negative coordinates" );
		throw_assert( _x < size.x && _y < size.y && _z < size.z, "Tried to read tensor out of bounds" );

		return data[
			_z * (size.x * size.y) +
				_y * (size.x) +
				_x
		];
	}

	

	void copy_from( std::vector<std::vector<std::vector<T>>> data )
	{
		int z = data.size();
		int y = data[0].size();
		int x = data[0][0].size();

		for ( int i = 0; i < x; i++ )
			for ( int j = 0; j < y; j++ )
				for ( int k = 0; k < z; k++ )
					get( i, j, k ) = data[k][j][i];
	}

	~tensor_t()
	{
		delete[] data;
	}

};

void randomize(tensor_t<float> & t, float max = 1.0) {
	for(int x = 0; x < t.size.x; x++) {
		for(int y = 0; y < t.size.y; y++) {
			for(int z = 0; z < t.size.z; z++) {
				t(x, y, z) = rand_f(max);
			}
		}
	}
}

void randomize(tensor_t<gradient_t> & t, float max = 1.0) {
	for(int x = 0; x < t.size.x; x++) {
		for(int y = 0; y < t.size.y; y++) {
			for(int z = 0; z < t.size.z; z++) {
				t(x, y, z).grad = rand_f(max);
				t(x, y, z).oldgrad = rand_f(max);
			}
		}
	}
}


template<class T>
std::ostream& operator<<(std::ostream& os, const tensor_t<T> & t)
{
	for ( int z = 0; z < t.size.z; z++ ) {
		os << z << ": \n";
		for ( int y = 0; y < t.size.y; y++ ) {
			for ( int x = 0; x < t.size.x; x++ ) {
				os << std::setw(2) << std::setprecision(2);
				os << t(x,y,z) << " ";
			}
			os << "\n";
		}
	}
	
	return os;
}

void print_tensor( tensor_t<float>& data )
{
	int mx = data.size.x;
	int my = data.size.y;
	int mz = data.size.z;

	for ( int z = 0; z < mz; z++ )
	{
		printf( "[Dim%d]\n", z );
		for ( int y = 0; y < my; y++ )
		{
			for ( int x = 0; x < mx; x++ )
			{
				printf( "%.2f \t", (float)data.get( x, y, z ) );
			}
			printf( "\n" );
		}
	}
}

tensor_t<float> to_tensor( std::vector<std::vector<std::vector<float>>> data )
{
	int z = data.size();
	int y = data[0].size();
	int x = data[0][0].size();


	tensor_t<float> t( x, y, z );

	for ( int i = 0; i < x; i++ )
		for ( int j = 0; j < y; j++ )
			for ( int k = 0; k < z; k++ )
				t( i, j, k ) = data[k][j][i];
	return t;
}


#ifdef INCLUDE_TESTS
#include "gtest/gtest.h"

namespace CNNTest {
	

       TEST_F(CNNTest, tensor_gradient) {
		tdsize s(2,2,3);
		tensor_t<gradient_t> t1(s);
		tensor_t<gradient_t> t2(s);

		EXPECT_EQ(t1, t2);
		randomize(t1, 1);
		EXPECT_NE(t1, t2);
       }

	TEST_F(CNNTest, tensor_operators) {
		tdsize s1(1,2,3);
		tdsize s2(2,3,4);

		EXPECT_EQ(s1, s1);
		EXPECT_NE(s1, s2);
		       
		tensor_t<float> t1(10, 10, 10);
		tensor_t<float> t2(10, 10, 10);
		tensor_t<float> t3(10, 10, 11);

		EXPECT_EQ(t1, t1);
		EXPECT_EQ(t1, t2);
		randomize(t1, 1);
		EXPECT_NE(t1, t2);

		EXPECT_NE(t1, t3);

		t3 = t1;
		EXPECT_EQ(t3, t1);
	}

}

#endif
