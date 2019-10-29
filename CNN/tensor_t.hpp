#pragma once
#include "types.hpp"
#include <vector>
#include <string.h>
#include <cmath>
#include <fstream>
#include <limits>

#define EPSILON 1e-8

static float rand_f(int maxval) {
	return 1.0f / maxval * rand() / float( RAND_MAX );
}

#define TDSIZE_FOR(T,X,Y,Z) for(int X = 0; X < T.x; X++) for(int Y = 0; Y < T.y; Y++) for(int Z = 0; Z < T.z; Z++) 
#define TENSOR_FOR(T,X,Y,Z) TDSIZE_FOR((T).size, X, Y, Z)

template<typename T>
struct tensor_t
{
	static const int version = 1;
	tdsize size;
	T * data;


	size_t calculate_data_size() const {
		return size.x *size.y *size.z * sizeof( T );
	}

	tensor_t( int _x, int _y, int _z ) :  size(_x, _y, _z) {
		throw_assert(size.x > 0 && size.y > 0 && size.z > 0,  "Tensor initialized with non-positive dimensions");
		data = new T[size.x * size.y * size.z]();
	}

	tensor_t(const tdsize & _size) : size(_size)
	{
		throw_assert(size.x > 0 && size.y > 0 && size.z > 0,  "Tensor initialized with non-positive dimensions");
		data = new T[size.x * size.y * size.z]();
	}

	tensor_t( const tensor_t& other ) : size(other.size)
	{
		data = new T[size.x *size.y *size.z];
		memcpy(
			data,
			other.data,
			calculate_data_size()
		);
	}

	tensor_t( tensor_t&& other ) : size(other.size), data(other.data)
	{
		other.data = nullptr;
	}

	~tensor_t()
	{
		delete[] data;
	}

	size_t get_total_memory_size() const {
		return calculate_data_size();
	}
	
	tensor_t<T> & operator=(const tensor_t& other )
	{
		if (&other != this) {
			delete[] data;
			size = other.size;
			data = new T[other.size.x *other.size.y *other.size.z];
			memcpy(
				this->data,
				other.data,
				calculate_data_size()
				);
		}
		return *this;
	}
	
	tensor_t<T> & operator=(tensor_t<T>&& other) {
		if (&other != this) {
			delete [] data;
			data = other.data;
			size = other.size;
			other.data = nullptr;
		}
		return *this;
	}
	
    
	tensor_t<T> operator+( const tensor_t<T>& other )const 
	{
		throw_assert(size == other.size, "Mismatche sizes is operator+");
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			clone.data[i] += other.data[i];
		return clone;
	}

	tensor_t<T> operator-( const tensor_t<T>& other ) const
	{

		throw_assert(size == other.size, "Mismatche sizes is operator-");
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z; i++ )
			clone.data[i] -= other.data[i];
		return clone;
	}
	
	inline T& operator()( int _x, int _y, int _z )
	{
		return this->get( _x, _y, _z );
	}

	inline const T& operator()( int _x, int _y, int _z ) const
	{
		return this->get( _x, _y, _z );
	}

	T& get( int _x, int _y, int _z ) {
		throw_assert_debug( _x >= 0 && _y >= 0 && _z >= 0, "Tried to read tensor at negative coordinates" );
		throw_assert_debug( _x < size.x && _y < size.y && _z < size.z, "Tried to read tensor out of bounds " << tdsize(_x, _y, _z) << ". But tensor is " << size );
		
		return data[
			_z * (size.x * size.y) +
			_y * (size.x) +
			_x
			];
	}

	const T & get( int _x, int _y, int _z ) const {
		throw_assert_debug( _x >= 0 && _y >= 0 && _z >= 0 , "Tried to read tensor at negative coordinates" );
		throw_assert_debug( _x < size.x && _y < size.y && _z < size.z, "Tried to read tensor out of bounds: read at " << tdsize(_x, _y, _z) << "; bound = " << size );
		
		return data[
			_z * (size.x * size.y) +
			_y * (size.x) +
			_x
			];
	}

	bool operator==(const tensor_t<T> & other) const
	{
		if (other.size != this->size)
			return false;

		TENSOR_FOR(*this, x,y,z) 
			if (other(x,y,z) != (*this)(x,y,z))
				return false;
		return true;
	}

	bool operator!=(const tensor_t<T> & o) const {
		return !(*this == o);
	}

	tensor_t<T> & paste(const tdsize & where, const tensor_t<T> & in,  bool grow=false) {
		if (grow) {
			throw_assert(false, "Grow not implemented");
		} else {
			throw_assert((where.x + in.size.x <= size.x) &&
				     (where.y + in.size.y <= size.y) &&
				     (where.z + in.size.z <= size.z), "Out of bounds tensor<>.copy_at()");
		}
		
		TDSIZE_FOR(in.size, x,y,z) 
			get(where.x + x,where.y + y,where.z + z) = in(x,y,z);
		return *this; 
	}

	tensor_t<T> copy(const tdsize & where, const tdsize & s,  bool grow=false) {
		if (grow) {
			throw_assert(false, "Grow not implemented");
		} else {
			throw_assert((where.x + s.x <= size.x) &&
				     (where.y + s.y <= size.y) &&
				     (where.z + s.z <= size.z),
				     "Out of bounds tensor<>.copy_at(). where = " << where << "; s = " << s << "; this->size = " << size);
		}

		tensor_t<T> n(s);
		
		TDSIZE_FOR(n.size, x,y,z) 
			n.get(x,y,z) = get(where.x + x,where.y + y,where.z + z);
		
		return n;
	}

	T max() const {
		auto l = argmax();
		return get(l.x,l.y,l.z);
	}
	
        T min() const {
		auto l = argmin();
		return get(l.x,l.y,l.z);
	}

	tensor_t<T> matmul(const tensor_t<T> & rhs) const {
		const tensor_t<T> & lhs = *this;
		throw_assert(lhs.size.y == rhs.size.x, "Matrix size mismatch in matmul: lhs = " << lhs.size << "; rhs = " << rhs.size);
		throw_assert(lhs.size.z == 1 && rhs.size.z == 1, "Matmul only works with depth-1 tensors: lhs = " << lhs.size << "; rhs = " << rhs.size);
		tensor_t<T> n(lhs.size.x, rhs.size.y, 1);
		TDSIZE_FOR(n.size, x,y,_) {
			float sum = 0;
			for(int i = 0; i < lhs.size.y; i++) {
				sum += lhs(x, i, 0) * rhs(i, y, 0);
			}
			n(x,y,_) = sum;
		}
		return n;
	}
	
	tdsize argmax() const {
		T max_value = -std::numeric_limits<float>::max();
		tdsize max_loc;
		
		TENSOR_FOR(*this, x,y,z) 
			if (get(x,y,z) > max_value) {
				max_value = get(x,y,z);
				max_loc = tdsize(x,y,z);
			}
		return max_loc;
	}
	tdsize argmin() const {
		T min_value = std::numeric_limits<float>::max();
		tdsize min_loc;
		TENSOR_FOR(*this, x,y,z) 
			if (get(x,y,z) < min_value) {
				min_value = get(x,y,z);
				min_loc = tdsize(x,y,z);
			}
		return min_loc;
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
	
	void write(std::ofstream & out) {
		out.write((char*)&version, sizeof(version));
		out.write((char*)&size, sizeof(size));
		out.write((char*)data, calculate_data_size());
	}

	static tensor_t<T> read(std::ifstream & in) {
		int version;
		tdsize size;
		in.read((char*)&version, sizeof(version));
		in.read((char*)&size, sizeof(size));
		tensor_t<T> n(size);
		throw_assert(version == n.version, "Reloading from old tensor version is not supported.  Current version: " << n.version << ";  file version: " << version);
		in.read((char*)n.data, n.calculate_data_size());
		return n;
	}

	
};

template<class T>
const int tensor_t<T>::version;


inline void randomize(tensor_t<float> & t, float max = 1.0) {
	TENSOR_FOR(t,x,y,z) {
		t(x, y, z) = rand_f(max);
	}
}

inline void randomize(tensor_t<gradient_t> & t, float max = 1.0) {
	TENSOR_FOR(t,x,y,z) {
		t(x, y, z).grad = rand_f(max);
		t(x, y, z).oldgrad = rand_f(max);
	}
}


template<class T>
static std::ostream& operator<<(std::ostream& os, const tensor_t<T> & t)
{
	for ( int z = 0; z < t.size.z; z++ ) {
		os << z << ": \n";
		for ( int y = 0; y < t.size.y; y++ ) {
			for ( int x = 0; x < t.size.x; x++ ) {
				os << std::setw(8) << std::setprecision(3);
				os << t(x,y,z) << " ";
			}
			os << "\n";
		}
	}
	
	return os;
}

template<class T>
static tensor_t<T> to_tensor( std::vector<std::vector<std::vector<T>>> data )
{
	int z = data.size();
	int y = data[0].size();
	int x = data[0][0].size();


	tensor_t<T> t( x, y, z );

	for ( int i = 0; i < x; i++ )
		for ( int j = 0; j < y; j++ )
			for ( int k = 0; k < z; k++ )
				t( i, j, k ) = data[k][j][i];
	return t;
}


#ifdef INCLUDE_TESTS
#include "gtest/gtest.h"


namespace CNNTest {


	TEST_F(CNNTest, tensor_matmul) {
		tensor_t<float> a(2,3,1), b(3,2,1);

		a(0,0,0) = 1;
		a(0,1,0) = 2;
		a(0,2,0) = 3;
		a(1,0,0) = 4;
		a(1,1,0) = 5;
		a(1,2,0) = 6;

		b(0,0,0) = 1;
		b(0,1,0) = 2;
		b(1,0,0) = 3;
		b(1,1,0) = 4;
		b(2,0,0) = 5;
		b(2,1,0) = 6;

		tensor_t<float> ab(2,2,1);
		ab(0,0,0) = 22;
		ab(0,1,0) = 28;
		ab(1,0,0) = 49;
		ab(1,1,0) = 64;

		EXPECT_EQ(ab, a.matmul(b));
		tensor_t<float> c(2,3,2), d(3,2,2);
		EXPECT_THROW(c.matmul(d), AssertionFailureException); // too "thick"

		tensor_t<float> f(2,4,1), g(3,2,1);
		EXPECT_THROW(f.matmul(g), AssertionFailureException); // mismatch dimensions.
		

	}
	
	TEST_F(CNNTest, tensor_for) {
		tensor_t<float> t1(3,4,5);
		float i = 0.0;
		TENSOR_FOR(t1, x, y, z) {
			t1(x,y,z) = i;
			i += 1.0;
		}
		float sum = 0.0;
		TENSOR_FOR(t1, x, y, z) {
			sum += t1(x,y,z);
		}
		EXPECT_EQ(sum, 1770.0);
	}

	TEST_F(CNNTest, tensor_slice) {
		tensor_t<float> t1(3,4,5);

		auto t2 = t1.copy({0,0,0}, {2, 3, 1});
		TDSIZE_FOR(tdsize(2,3,1), x,y,z)
			EXPECT_EQ(t1(x,y,z), t2(x,y,z));

		auto t3 = t1.copy({1,1,1}, {2, 3, 1});
		EXPECT_EQ(t3.size.x, 2);
		EXPECT_EQ(t3.size.y, 3);
		EXPECT_EQ(t3.size.z, 1);
		TDSIZE_FOR(tdsize(2,3,1), x,y,z)
			EXPECT_EQ(t1(x+1,y+1,z+1), t2(x,y,z));

		t1.paste({0,0,0}, t3);
		TDSIZE_FOR(t3.size, x,y,z)
			EXPECT_EQ(t1(x,y,z), t2(x,y,z));
		
	}
	
	TEST_F(CNNTest, tensor_gradient) {
		tdsize s(2,2,3);
		tensor_t<gradient_t> t1(s);
		tensor_t<gradient_t> t2(s);
		
		EXPECT_EQ(t1, t2);
		randomize(t1, 1);
		EXPECT_NE(t1, t2);

		EXPECT_EQ(t1.get_total_memory_size(), 2*2*3*sizeof(gradient_t));
	}
	
	TEST_F(CNNTest, tensor_io) {
		tensor_t<float> t1(11,14,23);
		randomize(t1);
		std::ofstream outfile (DEBUG_OUTPUT "t1_out.tensor",std::ofstream::binary);
		t1.write(outfile);
		outfile.close();
		std::ifstream infile (DEBUG_OUTPUT "t1_out.tensor",std::ofstream::binary);
		auto r = tensor_t<float>::read(infile);
		EXPECT_EQ(t1, r);

		tensor_t<gradient_t> t2(1,100,3);
		randomize(t2);
		std::ofstream outfile2 (DEBUG_OUTPUT "t2_out.tensor",std::ofstream::binary);
		t2.write(outfile2);
		outfile2.close();
		std::ifstream infile2 (DEBUG_OUTPUT "t2_out.tensor",std::ofstream::binary);
		auto r2 = tensor_t<gradient_t>::read(infile2);
		EXPECT_EQ(t2, r2);
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

		tensor_t<float> m(2,2,2);
		m(1,0,0) = -1;
		m(0,1,0) = 2;
		m(0,0,1) = 3;
		EXPECT_EQ(m.argmax(), tdsize(0,0,1));
		m(1,1,1) = 4;
		EXPECT_EQ(m.argmax(), tdsize(1,1,1));
		EXPECT_EQ(m.argmin(), tdsize(1,0,0));
		EXPECT_EQ(m.max(), 4);
		EXPECT_EQ(m.min(), -1);

		EXPECT_THROW(m - t1, AssertionFailureException); // mismatched sizes
		EXPECT_THROW(m + t1, AssertionFailureException); // mismatched sizes

		EXPECT_EQ(m.get_total_memory_size(), 2*2*2*sizeof(float));
		
	}

}

#endif
