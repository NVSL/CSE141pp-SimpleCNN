#pragma once
#include "types.hpp"
#include <vector>
#include <string.h>
#include <cmath>
#include <fstream>
#include <limits>

#include <gtest/gtest.h>
static float rand_f(int maxval) {
	return 1.0f / maxval * rand() / float( RAND_MAX );
}

// #define TDSIZE_FOR(T,X,Y,Z) for(int X = 0; X < T.x; X++) for(int Y = 0; Y < T.y; Y++) for(int Z = 0; Z < T.z; Z++)
#define TDSIZE_FOR(T,X,Y,Z,B) for(int X = 0; X < T.x; X++) for(int Y = 0; Y < T.y; Y++) for(int Z = 0; Z < T.z; Z++) for(int B = 0; B < T.b; B++)

// #define TENSOR_FOR(T,X,Y,Z) TDSIZE_FOR((T).size, X, Y, Z, 0)
#define TENSOR_FOR(T,X,Y,Z,B) TDSIZE_FOR((T).size, X, Y, Z, B)

#define EPSILON 1e-8

template<class T>
static bool almost_equal(T a, T b) {
        return std::abs(a-b) < EPSILON;
}
template<>
[[maybe_unused]] bool almost_equal(gradient_t a, gradient_t b) {
        return almost_equal(a.grad, b.grad) || almost_equal(a.oldgrad, b.oldgrad);
}

template<typename T>
struct tensor_t
{
	/* tensor_t is a 3D array of values.  It is the main input and
	   output type for the layers in our CNN models.  In almost
	   all cases, it holds floating point values of type float or
	   double.

	   The class has two key members: 
	   
	   - size : is of type tdsize and represents the size of the tensor (x,y,z)
	   
	   - data : data is a pointer to the data in the tensor.  

	   data is a liner (1D) array.

	   tensor_t provides two ways of accessing its contents. 

	   The first is via the () operator that lets you specify the
	   x, y, and z coordinates of the item you wish to access.
	   This is the most common access method. 

	   The code for the () operator (see below) defines how the
	   three dimensions map onto the `data` array.  This mapping
	   has important implications for how looping over the tensor
	   translates into memory accesses.

	   Alternately, you can call `as_vector()` to access it as a
	   linear array.  This method is mostly used in fc_layer_t.

	   The code for both of them is below.  Examine it carefully.
	*/

	tdsize size;
	T * data;
	
	T & as_vector(size_t i) {
		return data[i];
	}

	const  T & as_vector(size_t i) const {
		return data[i];
	}

	size_t element_count() const {
		return size.x * size.y * size.z * size.b;
	}

	void resize(tdsize new_size) {
		throw_assert(size.x > 0 && size.y > 0 && size.z > 0,  "Tensor resize with non-positive dimensions");
		size = new_size;
		delete[] data;
                if (size.b == 0) {
                        size.b = 1;
                }
                data = new T[size.x * size.y * size.z * size.b]();
	}

	inline void assert1D() const {
		throw_assert(
			     size.y == 1 &&
			     size.z == 1 &&
			     size.b == 1, "This matrix is not 1-dimensional")
	}
	inline void assert2D() const {
		throw_assert(
			     
			     size.z == 1 &&
			     size.b == 1, "This matrix is not 2-dimensional")
	}
	inline void assert3D() const {
		throw_assert(
			     size.b == 1, "This matrix is not 3-dimensional")
	}
	inline T& operator()( int _x, int _y, int _z, int _b=0 )
	{
		return this->get( _x, _y, _z, _b );
	}

	inline const T& operator()( int _x, int _y, int _z, int _b=0 ) const
	{
		return this->get( _x, _y, _z, _b );
	}

	T& get( int _x, int _y, int _z, int _b=0 ) {
		throw_assert_debug( _x >= 0 && _y >= 0 && _z >= 0 && _b >= 0, "Tried to read tensor at negative coordinates" );
		throw_assert_debug( _x < size.x && _y < size.y && _z < size.z && _b < size.b, "Tried to read tensor out of bounds " << tdsize(_x, _y, _z, _b) << ". But tensor is " << size );
		
		return data[
			_b * (size.x * size.y * size.z) +
			_z * (size.x * size.y) +
			_y * (size.x) +
			_x
			];
	}

	const T & get( int _x, int _y, int _z, int _b=0 ) const {
		throw_assert_debug( _x >= 0 && _y >= 0 && _z >= 0 && _b >= 0, "Tried to read tensor at negative coordinates" );
		throw_assert_debug( _x < size.x && _y < size.y && _z < size.z && _b < size.b, "Tried to read tensor out of bounds " << tdsize(_x, _y, _z, _b) << ". But tensor is " << size );
		
		return data[
			_b * (size.x * size.y * size.z) +
			_z * (size.x * size.y) +
			_y * (size.x) +
			_x
			];
	}

	
	static const int version = 1;
        static bool diff_prints_deltas;

	size_t calculate_data_size() const {
		return size.x * size.y * size.z * size.b * sizeof( T );
	}

	tensor_t( int _x, int _y, int _z, int _b=1 ) :  size(_x, _y, _z, _b) {
		throw_assert(size.x > 0 && size.y > 0 && size.z > 0 && size.b > 0,  "Tensor initialized with non-positive dimensions");
		data = new T[size.x * size.y * size.z * size.b]();
	}

	tensor_t(const tdsize & _size) : size(_size)
	{
		throw_assert(size.x > 0 && size.y > 0 && size.z > 0,  "Tensor initialized with non-positive dimensions");
		if (size.b == 0) {
			size.b = 1;
		}
		data = new T[size.x * size.y * size.z * size.b]();
		// std::cout << "Made new tensor with size: " << size << std::endl;
	}

	tensor_t( const tensor_t& other ) : size(other.size)
	{
		data = new T[size.x * size.y * size.z * size.b];
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
			data = new T[other.size.x * other.size.y * other.size.z * other.size.b];
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
		throw_assert(size == other.size, "Mismatched sizes is operator+");
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z * other.size.b; i++ )
			clone.data[i] += other.data[i];
		return clone;
	}

	tensor_t<T> operator-( const tensor_t<T>& other ) const
	{

		throw_assert(size == other.size, "Mismatchef sizes is operator-");
		tensor_t<T> clone( *this );
		for ( int i = 0; i < other.size.x * other.size.y * other.size.z * other.size.b; i++ )
			clone.data[i] -= other.data[i];
		return clone;
	}
	

	bool operator==(const tensor_t<T> & other) const
	{
		if (other.size != this->size)
			return false;

		TENSOR_FOR(*this, x,y,z,b) 
		        if (!almost_equal(other(x,y,z,b),(*this)(x,y,z,b)))
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
				     (where.z + in.size.z <= size.z) &&
				     (where.b + in.size.b <= size.b), "Out of bounds tensor<>.copy_at()");
		}
		
		TDSIZE_FOR(in.size, x,y,z,b) 
			get(where.x + x, where.y + y, where.z + z, where.b + b) = in(x,y,z,b);
		return *this; 
	}

	tensor_t<T> copy(const tdsize & where, const tdsize & s,  bool grow=false) {
		if (grow) {
			throw_assert(false, "Grow not implemented");
		} else {
			throw_assert((where.x + s.x <= size.x) &&
				     (where.y + s.y <= size.y) &&
				     (where.z + s.z <= size.z) &&
				     (where.b + s.b <= size.b),
				     "Out of bounds tensor<>.copy_at(). where = " << where << "; s = " << s << "; this->size = " << size);
		}

		tensor_t<T> n(s);
		
		TDSIZE_FOR(n.size, x,y,z,b) 
			n.get(x,y,z,b) = get(where.x + x, where.y + y, where.z + z, where.b + b);
		
		return n;
	}

	T max() const {
		auto l = argmax();
		return get(l.x,l.y,l.z,l.b);
	}
	
    T min() const {
		auto l = argmin();
		return get(l.x,l.y,l.z,l.b);
	}

	tensor_t<T> matmul(const tensor_t<T> & rhs) const {
		const tensor_t<T> & lhs = *this;
		throw_assert(lhs.size.y == rhs.size.x, "Matrix size mismatch in matmul: lhs = " << lhs.size << "; rhs = " << rhs.size);
		throw_assert(lhs.size.z == 1 && rhs.size.z == 1, "Matmul only works with depth-1 tensors: lhs = " << lhs.size << "; rhs = " << rhs.size);
		tensor_t<T> n(lhs.size.x, rhs.size.y, 1);
		TDSIZE_FOR(n.size, x,y,_,__) {
			double sum = 0;
			for(int i = 0; i < lhs.size.y; i++) {
				sum += lhs(x, i, 0) * rhs(i, y, 0);
			}
			n(x,y,_,__) = sum;
		}
		return n;
	}
	
	tdsize argmax() const {
		T max_value = -std::numeric_limits<double>::max();
		tdsize max_loc;
		
		TENSOR_FOR(*this, x,y,z,b) 
			if (get(x,y,z) > max_value) {
				max_value = get(x,y,z,b);
				max_loc = tdsize(x,y,z,b);
			}
		return max_loc;
	}
	tdsize argmin() const {
		T min_value = std::numeric_limits<double>::max();
		tdsize min_loc;
		TENSOR_FOR(*this, x,y,z,b) 
			if (get(x,y,z) < min_value) {
				min_value = get(x,y,z,b);
				min_loc = tdsize(x,y,z,b);
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

		// std::cout << "Reading size: " << sizeof(size) << std::endl;
		// std::cout << "Reading size: " << sizeof(int)*3 << std::endl;
		// in.read((char*)&size, sizeof(size));
		in.read((char*)&size, sizeof(size));
		tensor_t<T> n(size);
		// std::cout << "Reading with size: " << n << " " << size << std::endl;
		throw_assert(version == n.version, "Reloading from old tensor version is not supported.  Current version: " << n.version << ";  file version: " << version);
		in.read((char*)n.data, n.calculate_data_size());
		return n;
	}

	
};

template<class T>
const int tensor_t<T>::version;

template<class T>
bool tensor_t<T>::diff_prints_deltas = false;

inline void randomize(tensor_t<double> & t, double max = 1.0) {
	TENSOR_FOR(t,x,y,z, b) {
		t(x, y, z, b) = rand_f(max);
	}
}

inline void randomize(tensor_t<gradient_t> & t, double max = 1.0) {
	TENSOR_FOR(t,x,y,z,b) {
		t(x, y, z, b).grad = rand_f(max);
		t(x, y, z, b).oldgrad = rand_f(max);
	}
}


template<class T>
static std::ostream& operator<<(std::ostream& os, const tensor_t<T> & t)
{
	for ( int b = 0; b < t.size.b; b ++ ) {
		os << "b = " << b << ": \n";
		for ( int z = 0; z < t.size.z; z++ ) {
			os << "z = " << z << ": \n";
			for ( int y = 0; y < t.size.y; y++ ) {
				for ( int x = 0; x < t.size.x; x++ ) {
					os << std::setw(8) << std::setprecision(3);
					os << t(x,y,z,b) << " ";
				}
				os << "\n";
			}
		}
	}
	
	return os;
}

template<class T>
static tensor_t<T> to_tensor( std::vector<std::vector<std::vector<T>>> data )
{
	int b = 0;
	int z = data.size();
	int y = data[0].size();
	int x = data[0][0].size();


	tensor_t<T> t( x, y, z, b );

	for ( int i = 0; i < x; i++ )
		for ( int j = 0; j < y; j++ )
			for ( int k = 0; k < z; k++ )
				t( i, j, k ) = data[k][j][i];
	return t;
}


template<class T>
static std::string diff(const tensor_t<T> & first, const tensor_t<T> & second) 
{
	std::stringstream out;
	tensor_t<bool> diff(first.size);
	bool found = false;
    bool deltas = tensor_t<double>::diff_prints_deltas;

    for ( int b = 0; b < diff.size.b; b ++ ) {
		out << "b = " << b << ": \n";
		for ( int z = 0; z < diff.size.z; z++ ) {
			out << "z = " << z << ": \n";
			for ( int y = 0; y < diff.size.y; y++ ) {
				for ( int x = 0; x < diff.size.x; x++ ) {
				        if (!almost_equal(first(x,y,z), second(x,y,z))) found = true;
					if (deltas) {
					        out  << std::setprecision(2) << first(x,y,z,b) - second(x,y,z,b) << " ";
					} else {
	       				    out << (!almost_equal(first(x,y,z,b), second(x,y,z,b)) ? "#" : ".");
					}
				}
				out << "\n";
			}
		}
	}


	if (found) {
		return "\n" + out.str();
	} else {
		return "<identical>";
	}
	
}

template<class T>
static std::string diff(const std::vector<T> & a, const std::vector<T> & b)
{
	std::stringstream out;
	std::vector<bool> diff(a.size());
	bool found = false;
        bool deltas = tensor_t<double>::diff_prints_deltas;
	for ( uint x = 0; x < diff.size(); x++ ) {
	        if (!almost_equal(a[x], b[x])) found = true;
	        if (deltas) {
		       out << std::setprecision(2) <<  a[x] - b[x] << " ";
		} else {
     		       out << (!almost_equal(a[x], b[x]) ? "#" : ".");
		}
	}
	out << "\n";

	if (found) {
		return "\n" + out.str();
	} else {
		return "<identical>";
	}
	
}
template<>
[[maybe_unused]] std::string diff(const std::vector<gradient_t> & a, const std::vector<gradient_t> & b)
{
	std::stringstream out;
	std::vector<bool> diff(a.size());
	bool found = false;
        bool deltas = tensor_t<double>::diff_prints_deltas;
	for ( uint x = 0; x < diff.size(); x++ ) {
	        if (!almost_equal(a[x], b[x])) found = true;
	        if (deltas) {
		       out  << std::setprecision(2) << "<" << (a[x].grad - b[x].grad) << ", " << (a[x].oldgrad - b[x].oldgrad) << "> ";
		} else {
     		       out << (!almost_equal(a[x], b[x]) ? "#" : ".");
		}
	}
	out << "\n";

	if (found) {
		return "\n" + out.str();
	} else {
		return "<identical>";
	}
	
}

// Customized assertion formatter for googletest
template<class T>
::testing::AssertionResult AssertTensorsEqual(const char* m_expr,
					      const char* n_expr,
					      const tensor_t<T> & m,
					      const tensor_t<T> & n) {
	if (m == n) return ::testing::AssertionSuccess();

	return ::testing::AssertionFailure() << "Here's what's different. '#' denotes a position where your result is incorrect.\n" << diff(m, n);
}

#define ASSERT_TENSORS_EQ(T, a,b) ASSERT_PRED_FORMAT2(AssertTensorsEqual<T>, a,b)
#define EXPECT_TENSORS_EQ(T, a,b) EXPECT_PRED_FORMAT2(AssertTensorsEqual<T>, a,b)


#ifdef INCLUDE_TESTS
#include <gtest/gtest.h>

namespace CNNTest {


	TEST_F(CNNTest, tensor_matmul) {
		tensor_t<double> a(2,3,1), b(3,2,1);

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

		tensor_t<double> ab(2,2,1);
		ab(0,0,0) = 22;
		ab(0,1,0) = 28;
		ab(1,0,0) = 49;
		ab(1,1,0) = 64;

		EXPECT_EQ(ab, a.matmul(b));
		tensor_t<double> c(2,3,2), d(3,2,2);
		EXPECT_THROW(c.matmul(d), AssertionFailureException); // too "thick"

		tensor_t<double> f(2,4,1), g(3,2,1);
		EXPECT_THROW(f.matmul(g), AssertionFailureException); // mismatch dimensions.
		

	}
	
	TEST_F(CNNTest, tensor_for) {
		tensor_t<double> t1(3,4,5);
		double i = 0.0;
		TENSOR_FOR(t1, x, y, z, b) {
			t1(x,y,z) = i;
			i += 1.0;
		}
		double sum = 0.0;
		TENSOR_FOR(t1, x, y, z, b) {
			sum += t1(x,y,z);
		}
		EXPECT_EQ(sum, 1770.0);
	}

	TEST_F(CNNTest, tensor_slice) {
		tensor_t<double> t1(3,4,5);

		auto t2 = t1.copy({0,0,0}, {2, 3, 1});
		TDSIZE_FOR(tdsize(2,3,1), x,y,z,b)
			EXPECT_EQ(t1(x,y,z), t2(x,y,z));

		auto t3 = t1.copy({1,1,1}, {2, 3, 1});
		EXPECT_EQ(t3.size.x, 2);
		EXPECT_EQ(t3.size.y, 3);
		EXPECT_EQ(t3.size.z, 1);
		TDSIZE_FOR(tdsize(2,3,1), x,y,z,b)
			EXPECT_EQ(t1(x+1,y+1,z+1), t2(x,y,z));

		t1.paste({0,0,0}, t3);
		TDSIZE_FOR(t3.size, x,y,z,b)
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
		tensor_t<double> t1(11,14,23);
		randomize(t1);
		std::ofstream outfile (DEBUG_OUTPUT "t1_out.tensor",std::ofstream::binary);
		t1.write(outfile);
		outfile.close();
		std::ifstream infile (DEBUG_OUTPUT "t1_out.tensor",std::ofstream::binary);
		auto r = tensor_t<double>::read(infile);
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
		       
		tensor_t<double> t1(10, 10, 10);
		tensor_t<double> t2(10, 10, 10);
		tensor_t<double> t3(10, 10, 11);

		EXPECT_EQ(t1, t1);
		EXPECT_EQ(t1, t2);
		randomize(t1, 1);
		EXPECT_NE(t1, t2);

		EXPECT_NE(t1, t3);

		t3 = t1;
		EXPECT_EQ(t3, t1);

		tensor_t<double> m(2,2,2);
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

		EXPECT_EQ(m.get_total_memory_size(), 2*2*2*sizeof(double));
		
	}

}

#endif


