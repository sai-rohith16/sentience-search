#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

bool read_magic(std::ifstream& in) {
  char magic[6];
  in.read(magic, 6);
  return in.gcount() == 6 && std::memcmp(magic, "\x93NUMPY", 6) == 0;
}

std::string read_npy_header(std::ifstream& in, size_t& data_offset) {
  unsigned char ver[2];
  in.read(reinterpret_cast<char*>(ver), 2);
  if (in.gcount() != 2) {
    throw std::runtime_error("NPY: could not read version");
  }
  std::uint32_t header_len = 0;
  if (ver[0] == 1 && ver[1] == 0) {
    std::uint16_t len16 = 0;
    in.read(reinterpret_cast<char*>(&len16), 2);
    if (in.gcount() != 2) {
      throw std::runtime_error("NPY: could not read v1 header length");
    }
    header_len = len16;
  } else if (ver[0] == 2 && ver[1] == 0) {
    in.read(reinterpret_cast<char*>(&header_len), 4);
    if (in.gcount() != 4) {
      throw std::runtime_error("NPY: could not read v2 header length");
    }
  } else {
    throw std::runtime_error("NPY: unsupported format version");
  }

  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  if (static_cast<std::size_t>(in.gcount()) != header_len) {
    throw std::runtime_error("NPY: truncated header");
  }
  data_offset = static_cast<size_t>(in.tellg());
  return header;
}

void parse_descr_float32(const std::string& hdr) {
  auto pos = hdr.find("'descr'");
  if (pos == std::string::npos) {
    pos = hdr.find("\"descr\"");
  }
  if (pos == std::string::npos) {
    throw std::runtime_error("NPY: missing descr in header");
  }
  pos = hdr.find(':', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("NPY: malformed descr");
  }
  ++pos;
  while (pos < hdr.size() && (hdr[pos] == ' ' || hdr[pos] == '\t')) {
    ++pos;
  }
  char q = hdr[pos];
  if (q != '\'' && q != '"') {
    throw std::runtime_error("NPY: descr not quoted");
  }
  size_t start = pos + 1;
  size_t end = hdr.find(q, start);
  if (end == std::string::npos) {
    throw std::runtime_error("NPY: unterminated descr");
  }
  std::string descr = hdr.substr(start, end - start);
  if (descr != "<f4" && descr != "|f4") {
    throw std::runtime_error(
        "NPY: expected float32 descr '<f4' or '|f4', got: " + descr);
  }
}

std::pair<size_t, size_t> parse_shape_2d(const std::string& hdr) {
  auto pos = hdr.find("'shape'");
  if (pos == std::string::npos) {
    pos = hdr.find("\"shape\"");
  }
  if (pos == std::string::npos) {
    throw std::runtime_error("NPY: missing shape in header");
  }
  pos = hdr.find('(', pos);
  if (pos == std::string::npos) {
    throw std::runtime_error("NPY: shape tuple not found");
  }
  ++pos;
  size_t end = hdr.find(')', pos);
  if (end == std::string::npos) {
    throw std::runtime_error("NPY: unclosed shape tuple");
  }
  std::string inside = hdr.substr(pos, end - pos);
  std::vector<size_t> dims;
  size_t i = 0;
  while (i < inside.size()) {
    while (i < inside.size() &&
           (inside[i] == ' ' || inside[i] == '\t' || inside[i] == ',')) {
      ++i;
    }
    if (i >= inside.size()) {
      break;
    }
    char* p_end = nullptr;
    unsigned long long v = std::strtoull(inside.c_str() + i, &p_end, 10);
    if (p_end == inside.c_str() + i) {
      throw std::runtime_error("NPY: invalid shape integer");
    }
    dims.push_back(static_cast<size_t>(v));
    i = static_cast<size_t>(p_end - inside.c_str());
  }
  if (dims.size() != 2) {
    throw std::runtime_error("NPY: expected 2-D matrix, got " +
                             std::to_string(dims.size()) + " dimensions");
  }
  return {dims[0], dims[1]};
}

bool parse_fortran_order(const std::string& hdr) {
  auto pos = hdr.find("'fortran_order'");
  if (pos == std::string::npos) {
    pos = hdr.find("\"fortran_order\"");
  }
  if (pos == std::string::npos) {
    return false;
  }
  pos = hdr.find(':', pos);
  if (pos == std::string::npos) {
    return false;
  }
  ++pos;
  while (pos < hdr.size() && (hdr[pos] == ' ' || hdr[pos] == '\t')) {
    ++pos;
  }
  if (hdr.compare(pos, 4, "True") == 0) {
    return true;
  }
  return false;
}

}  // namespace

class SearchEngine {
 public:
  SearchEngine() = default;

  explicit SearchEngine(std::string path) { load(std::move(path)); }

  void load(std::string path) {
    path_ = std::move(path);
    std::ifstream in(path_, std::ios::binary);
    if (!in) {
      throw std::runtime_error("Failed to open file: " + path_);
    }
    if (!read_magic(in)) {
      throw std::runtime_error("Not a NumPy .npy file: " + path_);
    }
    size_t data_offset = 0;
    std::string header = read_npy_header(in, data_offset);
    if (parse_fortran_order(header)) {
      throw std::runtime_error(
          "NPY: Fortran-order arrays are not supported (use C-contiguous)");
    }
    parse_descr_float32(header);
    auto sh = parse_shape_2d(header);
    rows_ = sh.first;
    dim_ = sh.second;
    const size_t num_floats = rows_ * dim_;
    data_.resize(num_floats);
    in.seekg(static_cast<std::streamoff>(data_offset), std::ios::beg);
    in.read(reinterpret_cast<char*>(data_.data()),
            static_cast<std::streamsize>(num_floats * sizeof(float)));
    if (static_cast<size_t>(in.gcount()) != num_floats * sizeof(float)) {
      throw std::runtime_error("NPY: truncated data");
    }
  }

  [[nodiscard]] size_t num_movies() const { return rows_; }
  [[nodiscard]] size_t dimensions() const { return dim_; }

  // Returns indices of the k nearest rows by Euclidean distance (ascending).
  // Uses squared distance in a max-heap; ordering matches true Euclidean distance.
  [[nodiscard]] std::vector<int> search(const std::vector<float>& query,
                                        int k) const {
    if (rows_ == 0 || dim_ == 0) {
      throw std::runtime_error("SearchEngine: no vectors loaded");
    }
    if (k <= 0) {
      throw std::runtime_error("k must be positive");
    }
    if (query.size() != dim_) {
      throw std::runtime_error("query length " + std::to_string(query.size()) +
                               " does not match vector dimension " +
                               std::to_string(dim_));
    }
    const size_t kk = static_cast<size_t>(k);
    const size_t k_eff = std::min(kk, rows_);
    const float* qptr = query.data();

    using Entry = std::pair<float, size_t>;
    std::priority_queue<Entry> heap;

    for (size_t r = 0; r < rows_; ++r) {
      const float* row = data_.data() + r * dim_;
      float dist_sq = 0.f;
      for (size_t j = 0; j < dim_; ++j) {
        float d = row[j] - qptr[j];
        dist_sq += d * d;
      }
      if (heap.size() < k_eff) {
        heap.push({dist_sq, r});
      } else if (dist_sq < heap.top().first) {
        heap.pop();
        heap.push({dist_sq, r});
      }
    }

    std::vector<Entry> best;
    best.reserve(heap.size());
    while (!heap.empty()) {
      best.push_back(heap.top());
      heap.pop();
    }
    std::sort(best.begin(), best.end(), [](const Entry& a, const Entry& b) {
      if (a.first != b.first) {
        return a.first < b.first;
      }
      return a.second < b.second;
    });

    std::vector<int> out;
    out.reserve(best.size());
    for (const auto& e : best) {
      out.push_back(static_cast<int>(e.second));
    }
    return out;
  }

 private:
  std::string path_;
  std::vector<float> data_;
  size_t rows_ = 0;
  size_t dim_ = 0;
};

PYBIND11_MODULE(sentience_engine, m) {
  m.doc() = "KNN search over movie embedding vectors (Euclidean distance).";

  py::class_<SearchEngine>(m, "SearchEngine")
      .def(py::init<>())
      .def(py::init<std::string>(), py::arg("path"))
      .def("load", &SearchEngine::load, py::arg("path"))
      .def("num_movies", &SearchEngine::num_movies)
      .def("dimensions", &SearchEngine::dimensions)
      .def(
          "search",
          [](const SearchEngine& self, py::array_t<float> query, int k) {
            py::buffer_info qinfo = query.request();
            if (qinfo.ndim != 1 && qinfo.ndim != 2) {
              throw std::runtime_error(
                  "query must be 1-D or a single row (2-D)");
            }
            size_t q_len = 0;
            if (qinfo.ndim == 1) {
              q_len = static_cast<size_t>(qinfo.shape[0]);
            } else {
              if (qinfo.shape[0] != 1) {
                throw std::runtime_error(
                    "query with ndim=2 must have shape (1, dim)");
              }
              q_len = static_cast<size_t>(qinfo.shape[1]);
            }
            if (q_len != self.dimensions()) {
              throw std::runtime_error(
                  "query length " + std::to_string(q_len) +
                  " does not match vector dimension " +
                  std::to_string(self.dimensions()));
            }
            const auto* qptr = static_cast<const float*>(qinfo.ptr);
            std::vector<float> qvec(q_len);
            std::memcpy(qvec.data(), qptr, q_len * sizeof(float));
            std::vector<int> idx = self.search(qvec, k);
            py::array_t<std::int64_t> out(static_cast<py::ssize_t>(idx.size()));
            py::buffer_info oinfo = out.request();
            auto* optr = static_cast<std::int64_t*>(oinfo.ptr);
            for (size_t i = 0; i < idx.size(); ++i) {
              optr[i] = static_cast<std::int64_t>(idx[i]);
            }
            return out;
          },
          py::arg("query"), py::arg("k") = 5,
          "Return indices of the k nearest rows by Euclidean distance "
          "(smaller distance = more similar).");
}
