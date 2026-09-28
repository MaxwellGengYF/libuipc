#pragma once
/********************************************************************
 * @file export.h
 * @brief DLL export/import macro for the luisa_device layer.
 *
 * luisa_device has a few out-of-line definitions (the LC runtime singleton and
 * the LStream command-list facade) so that exactly one LC Context exists per
 * process even though the layer is consumed from both the backend DLL and the
 * backend tests. Those symbols therefore need an explicit DLL export.
 *
 * Define UIPC_LUISA_DEVICE_EXPORT=1 when compiling the definitions (the backend
 * target and the standalone test build); consumers get dllimport.
 *********************************************************************/
#if defined(_WIN32) || defined(_MSC_VER)
#if defined(UIPC_LUISA_DEVICE_EXPORT)
#define UIPC_LUISA_DEVICE_API __declspec(dllexport)
#else
#define UIPC_LUISA_DEVICE_API __declspec(dllimport)
#endif
#else
#define UIPC_LUISA_DEVICE_API
#endif
