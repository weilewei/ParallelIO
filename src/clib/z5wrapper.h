//
// Created by Kevin Paul on 2019-04-24.
//
#ifndef CZ5TEST_Z5WRAPPER_H
#define CZ5TEST_Z5WRAPPER_H

#include <stddef.h>
#ifdef __cplusplus
#include "z5/multiarray/xtensor_access.hxx"
#endif

#ifdef __cplusplus
#include "z5/dataset_factory.hxx"
#include "z5/file.hxx"
#include "z5/groups.hxx"
#include "z5/attributes.hxx"
#include "z5/compression/zlib_compressor.hxx"
#include "z5/types/types.hxx"
#endif

#include <stdint.h>

#ifdef __cplusplus
namespace z5 {
    extern "C" {
#endif
    void z5CreateFile(char* path);

    void z5CreateGroup(char* path);

    // float 32
    void z5CreateFloat32Dataset(char *path, unsigned int ndim, size_t *shape, size_t *chunks, int cuseZlib, int level);
    void z5WriteFloat32Subarray(char *path, float *array, unsigned int ndim, size_t *shape, size_t *offset);
    void z5ReadFloat32Subarray(char *path, float *array, unsigned int ndim, size_t *shape, size_t *offset);

    // float 64 / double
    void z5CreateFloat64Dataset(char *path, unsigned int ndim, size_t *shape, size_t *chunks, int cuseZlib, int level);
    void z5WriteFloat64Subarray(char *path, double *array, unsigned int ndim, size_t *shape, size_t *offset);
    void z5ReadFloat64Subarray(char *path, double *array, unsigned int ndim, size_t *shape, size_t *offset);

    // int8_t
    void z5CreateInt8Dataset(char *path, unsigned int ndim, size_t *shape, size_t *chunks, int cuseZlib, int level);
    void z5WriteInt8Subarray(char *path, int8_t *array, unsigned int ndim, size_t *shape, size_t *offset);
    void z5ReadInt8Subarray(char *path, int8_t *array, unsigned int ndim, size_t *shape, size_t *offset);

    // int16_t
    void z5CreateInt16Dataset(char *path, unsigned int ndim, size_t *shape, size_t *chunks, int cuseZlib, int level);
    void z5WriteInt16Subarray(char *path, int16_t *array, unsigned int ndim, size_t *shape, size_t *offset);
    void z5ReadInt16Subarray(char *path, int16_t *array, unsigned int ndim, size_t *shape, size_t *offset);

    // int32_t
    void z5CreateInt32Dataset(char *path, unsigned int ndim, size_t *shape, size_t *chunks, int cuseZlib, int level);
    void z5WriteInt32Subarray(char *path, int32_t *array, unsigned int ndim, size_t *shape, size_t *offset);
    void z5ReadInt32Subarray(char *path, int32_t *array, unsigned int ndim, size_t *shape, size_t *offset);

    // int64
    void z5CreateInt64Dataset(char *path, unsigned int ndim, size_t *shape, size_t *chunks, int cuseZlib, int level);
    void z5WriteInt64Subarray(char *path, long long int *array, unsigned int ndim, size_t *shape, size_t *offset);
    void z5ReadInt64Subarray(char *path, long long int *array, unsigned int ndim, size_t *shape, size_t *offset);

    // uint8_t
    void z5CreateUInt8Dataset(char *path, unsigned int ndim, size_t *shape, size_t *chunks, int cuseZlib, int level);
    void z5WriteUInt8Subarray(char *path, uint8_t *array, unsigned int ndim, size_t *shape, size_t *offset);
    void z5ReadUInt8Subarray(char *path, uint8_t *array, unsigned int ndim, size_t *shape, size_t *offset);

    // uint16_t
    void z5CreateUInt16Dataset(char *path, unsigned int ndim, size_t *shape, size_t *chunks, int cuseZlib, int level);
    void z5WriteUInt16Subarray(char *path, uint16_t *array, unsigned int ndim, size_t *shape, size_t *offset);
    void z5ReadUInt16Subarray(char *path, uint16_t *array, unsigned int ndim, size_t *shape, size_t *offset);

    // uint32_t
    void z5CreateUInt32Dataset(char *path, unsigned int ndim, size_t *shape, size_t *chunks, int cuseZlib, int level);
    void z5WriteUInt32Subarray(char *path, uint32_t *array, unsigned int ndim, size_t *shape, size_t *offset);
    void z5ReadUInt32Subarray(char *path, uint32_t *array, unsigned int ndim, size_t *shape, size_t *offset);

    // uint64
    void z5CreateUInt64Dataset(char *path, unsigned int ndim, size_t *shape, size_t *chunks, int cuseZlib, int level);
    void z5WriteUInt64Subarray(char *path, uint64_t *array, unsigned int ndim, size_t *shape, size_t *offset);
    void z5ReadUInt64Subarray(char *path, uint64_t *array, unsigned int ndim, size_t *shape, size_t *offset);

    size_t z5GetFileSize(char *path);

    void z5Delete(char *path );

    void z5writeAttributesString(char *path, const char *name, const char *value);

    void z5writeAttributesshort(char *path, const char *name, const short *value);

    void z5writeAttributesint(char *path, const char *name, const int *value);

    void z5writeAttributeslong(char *path, const char *name, const long *value);

    void z5writeAttributesfloat(char *path, const char *name, const float *value);

    void z5writeAttributesdouble(char *path, const char *name, const double *value);

    void z5writeAttributesushort(char *path, const char *name, const unsigned short *value);

    void z5writeAttributesusint(char *path, const char *name, const unsigned short int *value);

    void z5writeAttributeslonglong(char *path, const char *name, const long long *value);

    void z5writeAttributesulonglong(char *path, const char *name, const unsigned long long *value);

    void z5writeAttributesuint(char *path, const char *name, const unsigned int *value);

    // read attributes //

    void z5readAttributesWithKeys(char *path, char *keys[], int keys_sz);

    void z5readAttributes(char *path);
#ifdef __cplusplus
}
}
#endif
#endif //CZ5TEST_Z5WRAPPER_H
