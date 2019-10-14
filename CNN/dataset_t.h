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

	std::vector<test_case_t> test_cases;

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
		test_cases.push_back({data, label});
	}
	
	void add(const test_case_t & tc) {
		test_cases.push_back(tc);
	}
	
	void write(std::ofstream & out) {
		out.write((char*)&version, sizeof(version));
		size_t count = test_cases.size();
		out.write((char*)&count, sizeof(count));
		for(auto &c: test_cases) {
			c.write(out);
		}
	}

	static dataset_t read(std::ifstream & in) {
		int file_version;
		in.read((char*)&file_version, sizeof(version));
		throw_assert(version == file_version, "Reloading from old dataset version is not supported.  Current version: " << version << ";  file version: " << file_version);
		size_t count;
		in.read((char*)&count, sizeof(count));
		dataset_t n;
		for(uint i = 0; i < count; i++) {
			n.test_cases.push_back(test_case_t::read(in));
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
			test_case_t t {tensor_t<float>(2,2,2), tensor_t<float>(1,10,1)};
			ds.test_cases.push_back(t);
		}

		ds.get_total_memory_size();
		std::ofstream outfile2 ("ds_out.dataset",std::ofstream::binary);
		ds.write(outfile2);
		outfile2.close();
		std::ifstream infile2 ("ds_out.dataset",std::ofstream::binary);
		auto dsr = dataset_t::read(infile2);
		EXPECT_EQ(ds, dsr);
		
	}
}
#endif
