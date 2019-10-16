#pragma once

#include"tensor_t.h"
#include <fstream>

struct test_case_t
{
	static const int version = 1;
	
	tensor_t<float> data;
	tensor_t<float> label;

	size_t get_total_memory_size() const {
		return data.get_total_memory_size() + label.get_total_memory_size();
	}
	
	bool operator==(const test_case_t & other) const
	{
		return other.data == data && other.label == label;
	}

	bool operator!=(const test_case_t & o) const {
		return !(*this == o);
	}

	void write(std::ofstream & out) {
		out.write((char*)&version, sizeof(version));
		data.write(out);
		label.write(out);
	}

	
	static test_case_t read(std::ifstream & in) {
		int file_version;
		in.read((char*)&file_version, sizeof(version));
		throw_assert(version == file_version, "Reloading from old test_case version is not supported.  Current version: " << version << ";  file version: " << file_version);
		auto data = tensor_t<float>::read(in);
		auto label = tensor_t<float>::read(in);
		return {data, label};
	}

};

const int test_case_t::version; 

struct dataset_t
{
	static const int version = 1;
	tdsize data_size;
	tdsize label_size;

	std::vector<test_case_t> test_cases;
	typedef std::vector<test_case_t>::iterator iterator;
	typedef std::vector<test_case_t>::const_iterator const_iterator;
	
	size_t get_total_memory_size() const {
		size_t s = 0;
		for(auto &tc: test_cases) {
			s+= tc.get_total_memory_size();
		}
		return s;
	}
	
	bool operator==(const dataset_t & other) const
	{
		return other.test_cases == test_cases;
	}
	
	bool operator!=(const dataset_t & o) const {
		return !(*this == o);
	}
	
	void add(const tensor_t<float> & data, const tensor_t<float> & label) {
		add(test_case_t {data, label});
		//test_cases.push_back({data, label});
	}
	
	void add(const test_case_t & tc) {
		//throw_assert(tc.label.size.y == 1 &&
		//tc.label.size.z == 1, "Labels should have size (n,1,1).  Got size " << tc.label );

		if (test_cases.size() == 0) {
			data_size = tc.data.size;
			label_size = tc.label.size;
		} else {
			throw_assert(data_size == tc.data.size , "Test case data size doesn't match dataset. test case: " << tc.data.size << "; dataset: " << data_size);
			throw_assert(label_size == tc.label.size, "Test case label size doesn't match dataset. test case: " << tc.label.size << "; dataset: " << label_size);
		}
		test_cases.push_back(tc);
	}

	iterator begin() {
		return test_cases.begin();
	}
	
	iterator end() {
		return test_cases.end();
	}
	
	const_iterator begin() const {
		return test_cases.begin();
	}
	
	const_iterator end() const {
		return test_cases.end();
	}
	
	void write(std::ofstream & out) {
		out.write((char*)&version, sizeof(version));
		size_t count = test_cases.size();
		out.write((char*)&count, sizeof(count));
		for(auto &c: test_cases) {
			c.write(out);
		}
	}

	static dataset_t read(const std::string & s) {
		std::ifstream in(s,std::ofstream::binary);
		return dataset_t::read(in);
	}

	static dataset_t read(std::ifstream & in) {
		throw_assert(in.good(), "Input file descriptor in bad state");
		int file_version;
		in.read((char*)&file_version, sizeof(version));
		throw_assert(version == file_version, "Reloading from old dataset version is not supported.  Current version: " << version << ";  file version: " << file_version);
		size_t count;
		in.read((char*)&count, sizeof(count));
		dataset_t n;
		for(uint i = 0; i < count; i++) {
			n.add(test_case_t::read(in));
		}
		return n;
	}

};

const int dataset_t::version; 


#ifdef INCLUDE_TESTS
#include "gtest/gtest.h"


namespace CNNTest {

	TEST_F(CNNTest, dataset_io) {

		
		test_case_t t {tensor_t<float>(2,2,2), tensor_t<float>(1,10,1)};
		randomize(t.data);
		randomize(t.label);
		std::ofstream outfile ("t1_out.test_case",std::ofstream::binary);
		t.write(outfile);
		outfile.close();
		std::ifstream infile ("t1_out.test_case",std::ofstream::binary);
		auto r = test_case_t::read(infile);
		EXPECT_EQ(t, r);

		dataset_t ds;
		for(int i = 0; i < 11;i++){
			test_case_t t {tensor_t<float>(2,2,2), tensor_t<float>(10,1,1)};
			ds.add(t);
		}

		ds.get_total_memory_size();
		std::ofstream outfile2 ("ds_out.dataset",std::ofstream::binary);
		ds.write(outfile2);
		outfile2.close();
		std::ifstream infile2 ("ds_out.dataset",std::ofstream::binary);
		auto dsr = dataset_t::read(infile2);
		EXPECT_EQ(ds, dsr);

		
		EXPECT_THROW(ds.add(test_case_t {tensor_t<float>(3,3,3), tensor_t<float>(1,10,1)}), AssertionFailureException);
	}
}
#endif
