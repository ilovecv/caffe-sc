#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
        buf.size()));
      if (datum->label_size() > 0) {
       datum->set_label(0, label);
     } else {
       datum->add_label(label);
     }
     datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    //datum->set_label(label);
    if (datum->label_size() > 0) {
     datum->set_label(0, label);
   } else {
     datum->add_label(label);
   }
    return true;
  } else {
    return false;
  }
}

bool ReadImageToDatum(const string& filename, const std::vector<int> labels,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
        matchExt(filename, encoding) )
        return ReadFileToDatum(filename, labels, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
        buf.size()));
      for (int i = 0 ; i < labels.size(); ++i) {
       if (datum->label_size() <= i) {
         datum->add_label(labels[i]);
       } else {
         datum->set_label(i, labels[i]);
       }
     }
     datum->set_encoded(true);
     return true;
   }
   CVMatToDatum(cv_img, datum);
   for (int i = 0 ; i < labels.size(); ++i) {
     if (datum->label_size() <= i) {
       datum->add_label(labels[i]);
     } else {
       datum->set_label(i, labels[i]);
     }
   }
   return true;
 } else {
  return false;
  }
}

bool ReadImageToDatum(const string& filename1, const string& filename2, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img1 = ReadImageToCVMat(filename1, height, width, is_color);
  cv::Mat cv_img2 = ReadImageToCVMat(filename2, height, width, is_color);
  if (cv_img1.data && cv_img2.data) {
    if (encoding.size()) {
      std::vector<uchar> buf, buf_tmp;
      cv::imencode("."+encoding, cv_img1, buf);
      cv::imencode("."+encoding, cv_img2, buf_tmp);
      buf.insert(buf.end(), buf_tmp.begin(), buf_tmp.end());
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      //datum->set_label(label);
      if (datum->label_size() > 0) {
       datum->set_label(0, label);
     } else {
       datum->add_label(label);
     }
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img1, cv_img2, datum);
    //datum->set_label(label);
    if (datum->label_size() > 0) {
       datum->set_label(0, label);
     } else {
       datum->add_label(label);
     }
    return true;
  } else {
    return false;
  }
}

bool ReadImageToDatum(const string& filename1, const string& filename2, const std::vector<int> labels,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img1 = ReadImageToCVMat(filename1, height, width, is_color);
  cv::Mat cv_img2 = ReadImageToCVMat(filename2, height, width, is_color);
  if (cv_img1.data && cv_img2.data) {
    if (encoding.size()) {
      std::vector<uchar> buf, buf_tmp;
      cv::imencode("."+encoding, cv_img1, buf);
      cv::imencode("."+encoding, cv_img2, buf_tmp);
      buf.insert(buf.end(), buf_tmp.begin(), buf_tmp.end());
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      for (int i = 0 ; i < labels.size(); ++i) {
       if (datum->label_size() <= i) {
         datum->add_label(labels[i]);
       } else {
         datum->set_label(i, labels[i]);
       }
     }
     LOG(INFO) << "Label size is " << datum->label_size();
     datum->set_encoded(true);
     return true;
   }
   CVMatToDatum(cv_img1, cv_img2, datum);
   //LOG(INFO) << "Datum size: ";
   //LOG(INFO) << "channels = " << datum->channels();
   //LOG(INFO) << "height = " << datum->height();
   //LOG(INFO) << "width = " << datum->width();
   //LOG(INFO) << "Label size = " << labels.size();
   for (int i = 0 ; i < labels.size(); ++i) {
     if (datum->label_size() <= i) {
       datum->add_label(labels[i]);
     } else {
       datum->set_label(i, labels[i]);
     }
   }
     /*
    LOG(INFO) << "Expected label size is " << labels.size();
    LOG(INFO) << "Actuall label size is " << datum->label_size();
    LOG(INFO) << "And labels are ";
    for (int i = 0 ; i < datum->label_size(); ++i) {
    LOG(INFO) << datum->label(i);
    }
    LOG(INFO) << "Number of channels is " << datum->channels();
    */
   return true;
 } else {
  return false;
  }
}


bool ReadImageToDatum(const string& root_path, const std::vector<std::string> files,
    const std::vector<int> labels, const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  std::vector<cv::Mat> imgs;
  cv::Mat cv_img;
  bool img_empty = false;
  for (int i = 0; i < files.size(); i++){
      cv::Mat cv_img = ReadImageToCVMat(root_path+files[i], height, width, is_color);
      if(!cv_img.data){
        img_empty = true;
        break;
      }
      imgs.push_back(cv_img);
  }
  if (!img_empty) {
    if (encoding.size()) {
      std::vector<uchar> buf, buf_tmp;
      for (int i=0; i < files.size(); i++){
        cv::imencode("."+encoding, imgs[i], buf_tmp);
        buf.insert(buf.end(), buf_tmp.begin(), buf_tmp.end());
      }
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      for (int i = 0 ; i < labels.size(); ++i) {
       if (datum->label_size() <= i) {
         datum->add_label(labels[i]);
       } else {
         datum->set_label(i, labels[i]);
       }
     }
     LOG(INFO) << "Label size is " << datum->label_size();
     datum->set_encoded(true);
     return true;
   }
   CVMatToDatum(imgs, datum);
   //LOG(INFO) << "Datum size: ";
   //LOG(INFO) << "channels = " << datum->channels();
   //LOG(INFO) << "height = " << datum->height();
   //LOG(INFO) << "width = " << datum->width();
   //LOG(INFO) << "Label size = " << labels.size();
   for (int i = 0 ; i < labels.size(); ++i) {
     if (datum->label_size() <= i) {
       datum->add_label(labels[i]);
     } else {
       datum->set_label(i, labels[i]);
     }
   }

    /*LOG(INFO) << "Expected label size is " << labels.size();
    LOG(INFO) << "Actuall label size is " << datum->label_size();
    LOG(INFO) << "And labels are ";
    for (int i = 0 ; i < datum->label_size(); ++i) {
    LOG(INFO) << datum->label(i);
    }
    LOG(INFO) << "Number of channels is " << datum->channels();*/

   return true;
 } else {
  return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    if (datum->label_size() > 0) {
       datum->set_label(0, label);
     } else {
       datum->add_label(label);
     }
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}
bool ReadFileToDatum(const string& filename, const std::vector<int> labels,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    for (int i = 0 ; i < labels.size(); ++i) {
     if (datum->label_size() <= i) {
       datum->add_label(labels[i]);
     } else {
       datum->set_label(i, labels[i]);
     }
   }
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}
#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}void CVMatToDatum(const cv::Mat& cv_img1, const cv::Mat& cv_img2, Datum* datum) {
  CHECK(cv_img1.depth() == CV_8U && cv_img2.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img1.channels() * 2);
  datum->set_height(cv_img1.rows);
  datum->set_width(cv_img1.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  //int datum_channels = datum->channels();
  //int datum_height = datum->height();
  //int datum_width = datum->width();
  //int datum_size = datum_channels * datum_height * datum_width;
  //LOG(INFO) << "datum_size is " << datum_size;
  std::vector<uchar> buffer, vec_img_2;
  buffer.assign(cv_img1.datastart, cv_img1.dataend);
  vec_img_2.assign(cv_img2.datastart, cv_img2.dataend);
  buffer.insert(buffer.end(), vec_img_2.begin(), vec_img_2.end());
  datum->set_data(reinterpret_cast<uchar*>(&buffer[0]), buffer.size());
}


void CVMatToDatum(std::vector<cv::Mat> imgs, Datum* datum) {
  for (int i = 0; i < imgs.size(); i++){
    CHECK(imgs[i].depth() == CV_8U) << "Image data type must be unsigned byte";
  }
  datum->set_channels(imgs[0].channels() * imgs.size());
  datum->set_height(imgs[0].rows);
  datum->set_width(imgs[0].cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  //int datum_channels = datum->channels();
  //int datum_height = datum->height();
  //int datum_width = datum->width();
  //int datum_size = datum_channels * datum_height * datum_width;
  //LOG(INFO) << "datum_size is " << datum_size;
  std::vector<uchar> buffer, buf_tmp;
  for (int i = 0; i < imgs.size(); i++){
    buf_tmp.assign(imgs[i].datastart, imgs[i].dataend);
    buffer.insert(buffer.end(), buf_tmp.begin(), buf_tmp.end());
  }
  datum->set_data(reinterpret_cast<uchar*>(&buffer[0]), buffer.size());
  //LOG(INFO) << "Data size is " << datum->data().size();
}
#endif  // USE_OPENCV
}  // namespace caffe
