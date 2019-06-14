//
// Created by Kevin Paul on 2019-04-24.
//
#include <string>
#include <stdint.h>
#include <vector>
#include <iostream>
#include "z5wrapper.h"

namespace fs = boost::filesystem;
namespace z5 {
    extern "C" {

    void z5CreateFile(char* path)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::File cFile(path_s);
        createFile(cFile, asZarr);
    }

    void z5CreateGroup(char* path) {
        std::string path_s(path);
        bool asZarr = true;
        handle::Group cGroup(path_s);
        createGroup(cGroup, asZarr);
    }
    void z5CreateFloat32Dataset(char *path, unsigned int ndim, size_t *shape, size_t *count, int cuseZlib, int level) {
        std::string path_s(path);
        std::vector<std::string> dtype({"float32"});
        std::vector<size_t> shape_v(shape, shape + ndim);
        std::vector<size_t> count_v(count, count + ndim);
        bool asZarr = true;

        DatasetMetadata floatMeta(types::Datatype::float32,shape_v,count_v,asZarr);
        if (cuseZlib) {
            floatMeta.compressor = types::zlib;
            floatMeta.compressionOptions["useZlib"] = true;
            floatMeta.compressionOptions["level"] = level;
        }
        handle::Dataset handle_(path_s);
        handle_.createDir();
        writeMetadata(handle_, floatMeta);

    }

    void z5WriteFloat32Subarray(char *path, float *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds =openDataset(path_s);
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size=size*(*i);
        xt::xarray<float> adp_array=xt::adapt(array,size,xt::no_ownership(),count_v);
        std::vector<size_t> start_v(start,start + ndim);
        multiarray::writeSubarray<float>(ds,adp_array,start_v.begin());
    }

    void z5ReadFloat32Subarray(char *path, float *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds = openDataset(path_s);
        using vec_type = std::vector<float>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size*=(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::readSubarray<float>(ds,adp_array,start_v.begin());
    }

    void z5CreateFloat64Dataset(char *path, unsigned int ndim, size_t *shape, size_t *count, int cuseZlib, int level) {
        std::string path_s(path);
        std::vector<std::string> dtype({"float64"});
        std::vector<size_t> shape_v(shape, shape + ndim);
        std::vector<size_t> count_v(count, count + ndim);
        bool asZarr = true;

        DatasetMetadata floatMeta(types::Datatype::float64,shape_v,count_v,asZarr);
        if (cuseZlib) {
            floatMeta.compressor = types::zlib;
            floatMeta.compressionOptions["useZlib"] = true;
            floatMeta.compressionOptions["level"] = level;
        }
        handle::Dataset handle_(path_s);
        handle_.createDir();
        writeMetadata(handle_, floatMeta);

    }

    void z5WriteFloat64Subarray(char *path, double *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds =openDataset(path_s);
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size=size*(*i);
        xt::xarray<double> adp_array=xt::adapt(array,size,xt::no_ownership(),count_v);
        std::vector<size_t> start_v(start,start + ndim);
        multiarray::writeSubarray<double>(ds,adp_array,start_v.begin());
    }

    void z5ReadFloat64Subarray(char *path, double *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds = openDataset(path_s);
        using vec_type = std::vector<float>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size*=(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::readSubarray<double>(ds,adp_array,start_v.begin());
    }

    void z5CreateInt8Dataset(char *path, unsigned int ndim, size_t *shape, size_t *count, int cuseZlib, int level) {
        std::string path_s(path);

        std::vector<size_t> shape_v(shape, shape + ndim);
        std::vector<size_t> count_v(count, count + ndim);
        bool asZarr = true;

        DatasetMetadata int8Meta(types::Datatype::int8,shape_v,count_v,asZarr);
        if (cuseZlib) {
            int8Meta.compressor = types::zlib;
            int8Meta.compressionOptions["useZlib"] = true;
            int8Meta.compressionOptions["level"] = level;
        }
        handle::Dataset handle_(path_s);
        handle_.createDir();
        writeMetadata(handle_, int8Meta);
    }

    void z5WriteInt8Subarray(char *path, int8_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds =openDataset(path_s);
        using vec_type = std::vector<int8_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size=size*(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::writeSubarray<int8_t>(ds,adp_array,start_v.begin());
    }

    void z5ReadInt8Subarray(char *path, int8_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds = openDataset(path_s);
        using vec_type = std::vector<int8_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size*=(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::readSubarray<int8_t>(ds,adp_array,start_v.begin());
    }

    void z5CreateInt16Dataset(char *path, unsigned int ndim, size_t *shape, size_t *count, int cuseZlib, int level) {
        std::string path_s(path);

        std::vector<size_t> shape_v(shape, shape + ndim);
        std::vector<size_t> count_v(count, count + ndim);
        bool asZarr = true;

        DatasetMetadata int16Meta(types::Datatype::int16,shape_v,count_v,asZarr);
        if (cuseZlib) {
            int16Meta.compressor = types::zlib;
            int16Meta.compressionOptions["useZlib"] = true;
            int16Meta.compressionOptions["level"] = level;
        }
        handle::Dataset handle_(path_s);
        handle_.createDir();
        writeMetadata(handle_, int16Meta);
    }

    void z5WriteInt16Subarray(char *path, int16_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds =openDataset(path_s);
        using vec_type = std::vector<int16_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size=size*(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::writeSubarray<int16_t>(ds,adp_array,start_v.begin());
    }

    void z5ReadInt16Subarray(char *path, int16_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds = openDataset(path_s);
        using vec_type = std::vector<int16_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size*=(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::readSubarray<int16_t>(ds,adp_array,start_v.begin());
    }

    void z5CreateInt32Dataset(char *path, unsigned int ndim, size_t *shape, size_t *count, int cuseZlib, int level) {
        std::string path_s(path);

        std::vector<size_t> shape_v(shape, shape + ndim);
        std::vector<size_t> count_v(count, count + ndim);
        bool asZarr = true;

        DatasetMetadata int32Meta(types::Datatype::int32,shape_v,count_v,asZarr);
        if (cuseZlib) {
            int32Meta.compressor = types::zlib;
            int32Meta.compressionOptions["useZlib"] = true;
            int32Meta.compressionOptions["level"] = level;
        }
        handle::Dataset handle_(path_s);
        handle_.createDir();
        writeMetadata(handle_, int32Meta);
    }

    void z5WriteInt32Subarray(char *path, int32_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds =openDataset(path_s);
        using vec_type = std::vector<int32_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size=size*(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::writeSubarray<int32_t>(ds,adp_array,start_v.begin());
    }

    void z5ReadInt32Subarray(char *path, int32_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds = openDataset(path_s);
        using vec_type = std::vector<int32_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size*=(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::readSubarray<int32_t>(ds,adp_array,start_v.begin());
    }

    void z5CreateInt64Dataset(char *path, unsigned int ndim, size_t *shape, size_t *count, int cuseZlib, int level) {
        std::string path_s(path);

        std::vector<size_t> shape_v(shape, shape + ndim);
        std::vector<size_t> count_v(count, count + ndim);
        bool asZarr = true;

        DatasetMetadata int64Meta(types::Datatype::int64,shape_v,count_v,asZarr);
        if (cuseZlib) {
            int64Meta.compressor = types::zlib;
            int64Meta.compressionOptions["useZlib"] = true;
            int64Meta.compressionOptions["level"] = level;
        }
        handle::Dataset handle_(path_s);
        handle_.createDir();
        writeMetadata(handle_, int64Meta);
    }

    void z5WriteInt64Subarray(char *path, long long int *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds =openDataset(path_s);
        using vec_type = std::vector<long int>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size=size*(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::writeSubarray<long int>(ds,adp_array,start_v.begin());
    }

    void z5ReadInt64Subarray(char *path, long long int *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds = openDataset(path_s);
        using vec_type = std::vector<long int>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size*=(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::readSubarray<long int>(ds,adp_array,start_v.begin());
    }

    void z5CreateUInt8Dataset(char *path, unsigned int ndim, size_t *shape, size_t *count, int cuseZlib, int level) {
        std::string path_s(path);

        std::vector<size_t> shape_v(shape, shape + ndim);
        std::vector<size_t> count_v(count, count + ndim);
        bool asZarr = true;

        DatasetMetadata uint8Meta(types::Datatype::uint8,shape_v,count_v,asZarr);
        if (cuseZlib) {
            uint8Meta.compressor = types::zlib;
            uint8Meta.compressionOptions["useZlib"] = true;
            uint8Meta.compressionOptions["level"] = level;
        }
        handle::Dataset handle_(path_s);
        handle_.createDir();
        writeMetadata(handle_, uint8Meta);
    }

    void z5WriteUInt8Subarray(char *path, uint8_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds =openDataset(path_s);
        using vec_type = std::vector<uint8_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size=size*(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::writeSubarray<uint8_t>(ds,adp_array,start_v.begin());
    }

    void z5ReadUInt8Subarray(char *path, uint8_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds = openDataset(path_s);
        using vec_type = std::vector<uint8_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size*=(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::readSubarray<uint8_t>(ds,adp_array,start_v.begin());
    }

    void z5CreateUInt16Dataset(char *path, unsigned int ndim, size_t *shape, size_t *count, int cuseZlib, int level) {
        std::string path_s(path);

        std::vector<size_t> shape_v(shape, shape + ndim);
        std::vector<size_t> count_v(count, count + ndim);
        bool asZarr = true;

        DatasetMetadata uint16Meta(types::Datatype::uint16,shape_v,count_v,asZarr);
        if (cuseZlib) {
            uint16Meta.compressor = types::zlib;
            uint16Meta.compressionOptions["useZlib"] = true;
            uint16Meta.compressionOptions["level"] = level;
        }
        handle::Dataset handle_(path_s);
        handle_.createDir();
        writeMetadata(handle_, uint16Meta);
    }

    void z5WriteUInt16Subarray(char *path, uint16_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds =openDataset(path_s);
        using vec_type = std::vector<uint16_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size=size*(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::writeSubarray<uint16_t>(ds,adp_array,start_v.begin());
    }

    void z5ReadUInt16Subarray(char *path, uint16_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds = openDataset(path_s);
        using vec_type = std::vector<uint16_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size*=(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::readSubarray<uint16_t>(ds,adp_array,start_v.begin());
    }

    void z5CreateUInt32Dataset(char *path, unsigned int ndim, size_t *shape, size_t *count, int cuseZlib, int level) {
        std::string path_s(path);

        std::vector<size_t> shape_v(shape, shape + ndim);
        std::vector<size_t> count_v(count, count + ndim);
        bool asZarr = true;

        DatasetMetadata uint32Meta(types::Datatype::uint32,shape_v,count_v,asZarr);
        if (cuseZlib) {
            uint32Meta.compressor = types::zlib;
            uint32Meta.compressionOptions["useZlib"] = true;
            uint32Meta.compressionOptions["level"] = level;
        }
        handle::Dataset handle_(path_s);
        handle_.createDir();
        writeMetadata(handle_, uint32Meta);
    }

    void z5WriteUInt32Subarray(char *path, uint32_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds =openDataset(path_s);
        using vec_type = std::vector<uint32_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size=size*(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::writeSubarray<uint32_t>(ds,adp_array,start_v.begin());
    }

    void z5ReadUInt32Subarray(char *path, uint32_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds = openDataset(path_s);
        using vec_type = std::vector<uint32_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size*=(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::readSubarray<uint32_t>(ds,adp_array,start_v.begin());
    }

    void z5CreateUInt64Dataset(char *path, unsigned int ndim, size_t *shape, size_t *count, int cuseZlib, int level) {
        std::string path_s(path);

        std::vector<size_t> shape_v(shape, shape + ndim);
        std::vector<size_t> count_v(count, count + ndim);
        bool asZarr = true;

        DatasetMetadata uint64Meta(types::Datatype::uint64,shape_v,count_v,asZarr);
        if (cuseZlib) {
            uint64Meta.compressor = types::zlib;
            uint64Meta.compressionOptions["useZlib"] = true;
            uint64Meta.compressionOptions["level"] = level;
        }
        handle::Dataset handle_(path_s);
        handle_.createDir();
        writeMetadata(handle_, uint64Meta);
    }

    void z5WriteUInt64Subarray(char *path, uint64_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds =openDataset(path_s);
        using vec_type = std::vector<uint64_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size=size*(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::writeSubarray<uint64_t>(ds,adp_array,start_v.begin());
    }

    void z5ReadUInt64Subarray(char *path, uint64_t *array, unsigned int ndim, size_t *count, size_t *start) {
        std::string path_s(path);
        auto ds = openDataset(path_s);
        using vec_type = std::vector<uint64_t>;
        size_t size = 1;
        std::vector<std::size_t> count_v(count,count + ndim);
        for (std::vector<size_t>::const_iterator i = count_v.begin(); i != count_v.end(); ++i)
            size*=(*i);
        using count_type = std::vector<vec_type::size_type>;
        count_type s(count,count+ndim);
        std::vector<size_t> start_v(start,start + ndim);
        auto adp_array=xt::adapt(array,size,xt::no_ownership(),s);
        multiarray::readSubarray<uint64_t>(ds,adp_array,start_v.begin());
    }

    size_t z5GetFileSize(char *path){
        std::string path_s(path);
        fs::ifstream file(path_s, std::ios::binary);
        file.seekg(0, std::ios::end);
        const std::size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        file.close();
        return fileSize;
    }

    void z5writeAttributesString(char *path, const char *name, const char *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        std::string value_s(value);
        j[name_s] = value_s;
        writeAttributes(cHandle, j);
    }

    void z5writeAttributesshort(char *path, const char *name, const short *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        j[name_s] = (int64_t) *value;
        writeAttributes(cHandle, j);
    }

    void z5writeAttributesint(char *path, const char *name, const int *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        j[name_s] = (int64_t) *value;
        writeAttributes(cHandle, j);
    }

    void z5writeAttributeslong(char *path, const char *name, const long *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        j[name_s] = (int64_t) *value;
        writeAttributes(cHandle, j);
    }

    void z5writeAttributesfloat(char *path, const char *name, const float *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        j[name_s] = (double) *value;
        writeAttributes(cHandle, j);
    }

    void z5writeAttributesdouble(char *path, const char *name, const double *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        j[name_s] = *value;
        writeAttributes(cHandle, j);
    }

    void z5writeAttributesushort(char *path, const char *name, const unsigned short *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        j[name_s] = (int64_t) *value;
        writeAttributes(cHandle, j);
    }

    void z5writeAttributesusint(char *path, const char *name, const unsigned short *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        j[name_s] = (int64_t) *value;
        writeAttributes(cHandle, j);
    }

    void z5writeAttributeslonglong(char *path, const char *name, const long long *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        j[name_s] = (int64_t) *value;
        writeAttributes(cHandle, j);
    }

    void z5writeAttributesulonglong(char *path, const char *name, const unsigned long long *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        j[name_s] = (int64_t) *value;
        writeAttributes(cHandle, j);
    }

    void z5writeAttributesuint(char *path, const char *name, const unsigned int *value)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::string name_s(name);
        j[name_s] = (int64_t) *value;
        writeAttributes(cHandle, j);
    }

    void z5Delete(char *path ){
        std::string path_s(path);
        fs::path filename(path_s);
        fs::remove_all(filename);
    }

    // read attributes //
    void z5readAttributesWithKeys(char *path, char *keys[], int keys_sz)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        std::vector<std::string> keys_s;
        for (size_t i = 0; i < keys_sz; i++)
        {
            std::string keys_tmp(keys[i]);
            keys_s.push_back(keys_tmp);
        }
        readAttributes(cHandle, keys_s, j);
#ifdef _JASON_OUTPUT_
        for (auto it = j.begin(); it != j.end(); ++it)
	{
    		std::cout << "key: " << it.key() << ", value:" << it.value() << '\n';
	}
#endif
    }

    void z5readAttributes(char *path)
    {
        std::string path_s(path);
        bool asZarr = true;
        handle::Handle cHandle(path_s);
        nlohmann::json j;
        readAttributes(cHandle, j);
#ifdef _JASON_OUTPUT_
        for (auto it = j.begin(); it != j.end(); ++it)
	{
    		std::cout << "key: " << it.key() << ", value:" << it.value() << '\n';
	}
#endif
    }
    }

}
