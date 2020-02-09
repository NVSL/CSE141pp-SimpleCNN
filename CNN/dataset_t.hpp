#pragma once

#include"tensor_t.hpp"
#include <fstream>

// test_case_t holds an input and it's label, both as tensors.
struct test_case_t
{
	enum {VERSION = 1};
	tensor_t<double> data;
	tensor_t<double> label;

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
		int v = VERSION;
		out.write((char*)&v, sizeof(v));
		data.write(out);
		label.write(out);
	}

	
	static test_case_t read(std::ifstream & in) {
		int file_version;
		in.read((char*)&file_version, sizeof(file_version));
		throw_assert(VERSION == file_version, "Reloading from old test_case version is not supported.  Current version: " << VERSION << ";  file version: " << file_version);
		auto data = tensor_t<double>::read(in);
		auto label = tensor_t<double>::read(in);
		return {data, label};
	}

};


// dataset_t holds an array of test_case_t objects and provides the
// means to iterate over them.
struct dataset_t
{
    enum {VERSION = 1};
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

	size_t size() const {
		return test_cases.size();
	}
	void add(const tensor_t<double> & data, const tensor_t<double> & label) {
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
		int v = VERSION;
		out.write((char*)&v, sizeof(v));
		size_t count = test_cases.size();
		out.write((char*)&count, sizeof(count));
		for(auto &c: test_cases) {
			c.write(out);
		}
	}

	dataset_t batched_copy(int new_batch_size) {
		throw_assert(data_size.b==1, "Trying to batch an already batched dataset.");
		dataset_t n;

		// new sizes
		tdsize new_data_size = data_size;
		new_data_size.b = new_batch_size;

		//tdsize new_label_size = label_size;
		//new_label_size.b = new_batch_size;

		// batches
		int batch_index = 0;
		for (auto& t : test_cases ) {
			tensor_t<double>* batch_data = new tensor_t<double>(new_data_size);
			tensor_t<double>* batch_label = new tensor_t<double>(new_data_size);
			for (int x = 0; x < data_size.x; x++ )
				for (int y = 0; y < data_size.y; y++ )
					for (int z = 0; z < data_size.z; z++ )
						(*batch_data)(x, y, z, batch_index) = t.data(x, y, z);

			for (int x = 0; x < label_size.x; x++ )
				for (int y = 0; y < label_size.y; y++ )
					for (int z = 0; z < label_size.z; z++ )
						(*batch_label)(x, y, z, batch_index) = t.label(x, y, z);

			batch_index += 1;

			if (batch_index >= new_batch_size) {
				n.add(*batch_data, *batch_label);
				batch_index = 0;
			}
		}

		return n;

	}

	static dataset_t read(const std::string & s, std::vector<test_case_t>::size_type max_count = std::numeric_limits<std::vector<test_case_t>::size_type>::max()) {
		std::ifstream in(s,std::ofstream::binary);
		throw_assert(in.good(), "Couldn't open " << s);
		return dataset_t::read(in, max_count);
	}

	static dataset_t read(std::ifstream & in, std::vector<test_case_t>::size_type max_count = std::numeric_limits<std::vector<test_case_t>::size_type>::max()) {
		throw_assert(in.good(), "Input file descriptor in bad state");
		int file_version;
		in.read((char*)&file_version, sizeof(file_version));
		throw_assert(VERSION == file_version, "Reloading from old dataset version is not supported.  Current version: " << VERSION << ";  file version: " << file_version);
		size_t count;
		in.read((char*)&count, sizeof(count));
		dataset_t n;
		for(uint i = 0; i < count; i++) {
			n.add(test_case_t::read(in));
			if (n.test_cases.size() >= max_count) {
				break;
			}
		}
		return n;
	}

};


#ifdef INCLUDE_TESTS


namespace CNNTest {

	TEST_F(CNNTest, dataset_io) {

		
		test_case_t t {tensor_t<double>(2,2,2), tensor_t<double>(1,10,1)};
		randomize(t.data);
		randomize(t.label);
		std::ofstream outfile (DEBUG_OUTPUT "t1_out.test_case",std::ofstream::binary);
		t.write(outfile);
		outfile.close();
		std::ifstream infile (DEBUG_OUTPUT "t1_out.test_case",std::ofstream::binary);
		auto r = test_case_t::read(infile);
		EXPECT_EQ(t, r);

		dataset_t ds;
		for(int i = 0; i < 11;i++){
			test_case_t t {tensor_t<double>(2,2,2), tensor_t<double>(10,1,1)};
			ds.add(t);
		}

		ds.get_total_memory_size();
		std::ofstream outfile2 (DEBUG_OUTPUT "ds_out.dataset",std::ofstream::binary);
		ds.write(outfile2);
		outfile2.close();
		std::ifstream infile2 (DEBUG_OUTPUT "ds_out.dataset",std::ofstream::binary);
		auto dsr = dataset_t::read(infile2);
		EXPECT_EQ(ds, dsr);

		
		EXPECT_THROW(ds.add(test_case_t {tensor_t<double>(3,3,3), tensor_t<double>(1,10,1)}), AssertionFailureException);
	}
}
#endif

